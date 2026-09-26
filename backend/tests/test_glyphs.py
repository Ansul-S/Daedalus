import pytest

np = pytest.importorskip("numpy")  # the glyphs group; the tests that need more ask for it

from scripts import glyphs  # noqa: E402
from scripts.glyphs import (  # noqa: E402
    COVERAGE,
    LETTERS,
    RAMP,
    Picture,
    Spec,
    build,
    encode,
    levels,
    luminance,
    s_curve,
    write_module,
)


def test_the_ramp_runs_from_paper_to_ink() -> None:
    assert len(RAMP) == len(COVERAGE) == len(LETTERS) == 13
    assert RAMP[0][0] == " " and COVERAGE[0] == 0
    assert np.all(np.diff(COVERAGE) > 0)
    # the darkest steps are the reversed ones, cut out of a solid cell
    assert [reversed_ for _, _, reversed_ in RAMP] == [False] * 9 + [True] * 4


def test_paper_is_the_empty_step_and_ink_the_darkest() -> None:
    darkness = np.linspace(0, 1, 101).reshape(1, -1)

    steps = levels(darkness, floor=0.1)[0]

    assert steps[0] == 0 and steps[10] == 0  # at or below the floor: bare paper
    assert steps[-1] == 12
    assert np.all(np.diff(steps) >= 0)
    assert set(steps.tolist()) == set(range(13))


def test_the_s_curve_keeps_its_ends() -> None:
    darkness = np.array([0.0, 0.3, 0.5, 1.0])

    curved = s_curve(darkness, mid=0.5, k=6)

    assert curved[0] == 0 and np.isclose(curved[-1], 1) and np.isclose(curved[2], 0.5)
    assert curved[1] < 0.3  # darker than the midpoint pushes further from it


def test_grids_are_run_length_encoded_row_by_row() -> None:
    grid = np.array([[0, 0, 0, 12], [12, 5, 5, 7]])

    assert encode(grid) == "a3m2f2h"


def test_the_module_holds_the_ramp_and_every_grid(tmp_path) -> None:
    path = tmp_path / "grids.ts"

    write_module({"tiny": {"cols": 2, "rows": 1, "data": "am"}}, path)

    module = path.read_text("utf-8")
    assert '"tiny": { cols: 2, rows: 1, data: "am" },' in module
    assert '{"glyph": "Θ", "weight": 700, "reversed": true, "coverage": 0.5792}' in module
    assert "export type GridName = keyof typeof GRIDS;" in module


def test_a_crop_box_lands_on_the_same_spot_in_a_bigger_copy(tmp_path) -> None:
    image = pytest.importorskip("PIL.Image")
    # the box was measured on a 50 x 100 copy; this one is twice the size, inked where it falls
    pixels = np.full((200, 100, 3), 255, np.uint8)
    pixels[40:80, 20:60] = 0
    path = tmp_path / "copy.png"
    image.fromarray(pixels).save(path)

    lum = luminance(path, (10, 20, 30, 40), size=(50, 100))

    assert lum.shape == (40, 40)
    assert lum.max() == 0  # nothing but ink: the box grew with the copy


def test_each_picture_is_drawn_into_its_own_grids(tmp_path, monkeypatch) -> None:
    image = pytest.importorskip("PIL.Image")
    pytest.importorskip("cv2")
    # two made-up pictures: one inked on its left half, the other on its right
    left = np.full((60, 80, 3), 255, np.uint8)
    left[:, :40] = 0
    paths = {}
    for name, pixels in {"left": left, "right": left[:, ::-1].copy()}.items():
        paths[name] = tmp_path / f"{name}.png"
        image.fromarray(pixels).save(paths[name])

    def spec(cols: int) -> Spec:
        return Spec((0, 0, 80, 60), cols, 0.5, 1.0, 2, 50, 6, 0.10)

    monkeypatch.setattr(
        glyphs,
        "PICTURES",
        {
            "left": Picture("left", (80, 60), {"left": spec(8), "left-coarse": spec(4)}),
            "right": Picture("right", (80, 60), {"right": spec(8)}),
        },
    )

    grids = build(paths)

    assert list(grids) == ["left", "left-coarse", "right"]
    # rows keep the picture's shape in cells 0.6 as wide as tall
    assert [(grid["cols"], grid["rows"]) for grid in grids.values()] == [(8, 4), (4, 2), (8, 4)]
    # each grid's first cell is its own picture's: ink (a reversed step) in one, paper in the other
    assert RAMP[LETTERS.index(grids["left"]["data"][0])][2]
    assert grids["right"]["data"].startswith("a")


def test_the_module_credits_the_pictures_its_grids_come_from(tmp_path) -> None:
    path = tmp_path / "grids.ts"
    tiny = {"cols": 1, "rows": 1, "data": "a"}

    write_module({"daedalus": tiny, "minotaur-coarse": tiny}, path)

    module = path.read_text("utf-8")
    assert "//   daedalus: Charles Holroyd, Daedalus (1895)" in module
    assert "//   minotaur-coarse: Antonio Tempesta, Theseus and the Minotaur (after 1606)" in module

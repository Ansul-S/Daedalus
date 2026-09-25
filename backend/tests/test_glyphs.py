import pytest

np = pytest.importorskip("numpy")  # the glyphs group; OpenCV and Pillow aren't needed here

from scripts.glyphs import (  # noqa: E402
    COVERAGE,
    LETTERS,
    RAMP,
    encode,
    levels,
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

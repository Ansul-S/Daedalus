"""Redraw an etching in Greek letters: the grids behind the frontend's glyph pictures.

Each cell of a monospace grid (0.6 em wide, 1 em tall) takes one of 13 steps from paper to ink,
and the frontend draws each step as a letter; the four darkest are solid cells with the letter
cut out. Only the grids ship, run-length encoded: never the picture itself.

The picture is Charles Holroyd's etching *Daedalus* (1895, British Museum 1918,0608.347), in the
public domain. The crop boxes below were measured on a 736 x 942 copy of it.

Run from backend/:  uv run --group glyphs python -m scripts.glyphs path/to/daedalus.jpeg
"""

import argparse
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np

from app.core.config import REPO_ROOT

OUTPUT = REPO_ROOT / "frontend" / "src" / "lib" / "glyph" / "grids.ts"

# The copy the crop boxes were measured on
SOURCE_SIZE = (736, 942)

# A cell's width over its height: JetBrains Mono advances 0.6 em, and lines are 1 em apart.
CELL = 0.6

# From paper to ink: (letter, weight, reversed). A reversed step is a solid cell with the
# letter cut out of it, as red-figure painters left the figure in the clay and blacked in
# the ground; that gives the pictures a true black.
RAMP = [
    (" ", 400, False),
    ("·", 400, False),
    (":", 400, False),
    ("τ", 400, False),
    ("ε", 400, False),
    ("α", 400, False),
    ("θ", 400, False),
    ("Δ", 700, False),
    ("Ψ", 700, False),
    ("Θ", 700, True),
    ("σ", 700, True),
    ("τ", 400, True),
    (":", 400, True),
]
# How much of its cell each step inks, measured once by rendering the letters in a monospace
# font. Fixed here, so rebuilding a grid doesn't depend on which fonts a machine has.
COVERAGE = np.array(
    [0.0, 0.0308, 0.0931, 0.1447, 0.1876, 0.2376, 0.3072, 0.3683, 0.4396, 0.5792, 0.6863]
    + [0.8553, 0.9069]
)
# Step 0 to 12 in an encoded grid
LETTERS = "abcdefghijklm"


@dataclass(frozen=True)
class Spec:
    box: tuple[int, int, int, int]  # crop (left, top, right, bottom) in the 736 x 942 copy
    cols: int
    blur: float  # Gaussian blur, in cells: turns the etched hatching into tone
    clahe: float  # local contrast (CLAHE clip limit)
    tiles: int  # CLAHE tiles across and down
    mid: float  # the S-curve's centre, as a percentile of darkness
    k: float  # the S-curve's steepness
    floor: float  # darkness below this is bare paper


PLATE = (40, 30, 698, 902)
SPECS = {
    # The whole plate, for the landing page, and a coarser grid for small screens
    "daedalus": Spec(PLATE, 168, 0.7, 1.4, 6, 55, 6, 0.10),
    "daedalus-coarse": Spec(PLATE, 104, 0.7, 1.4, 6, 55, 6, 0.10),
    # Daedalus at his bench: the plate's upper right
    "thinker": Spec((210, 30, 698, 396), 96, 0.7, 1.3, 4, 55, 6, 0.10),
}


def luminance(image: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    """The crop's luminance, from 0 (ink) to 1 (paper)."""
    from PIL import Image

    with Image.open(image) as picture:
        scale = picture.width / SOURCE_SIZE[0]
        crop = tuple(round(edge * scale) for edge in box)
        rgb = np.asarray(picture.convert("RGB").crop(crop)).astype(np.float32) / 255
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def s_curve(darkness: np.ndarray, mid: float, k: float) -> np.ndarray:
    """Sigmoid contrast around `mid`, stretched back to 0-1."""

    def sigmoid(x):
        return 1 / (1 + np.exp(-k * (x - mid)))

    low, high = sigmoid(0.0), sigmoid(1.0)
    return (sigmoid(darkness) - low) / (high - low)


def tone(lum: np.ndarray, spec: Spec) -> np.ndarray:
    """Darkness per cell, 0 (paper) to 1 (ink)."""
    import cv2

    height, width = lum.shape
    rows = round(spec.cols * (height / width) * CELL)
    sigma = spec.blur * (width / spec.cols)
    blurred = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    small = cv2.resize(blurred, (spec.cols, rows), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(small, 1), np.percentile(small, 99.5)
    small = np.clip((small - low) / (high - low + 1e-6), 0, 1)
    clahe = cv2.createCLAHE(clipLimit=spec.clahe, tileGridSize=(spec.tiles, spec.tiles))
    small = clahe.apply(np.clip(small * 255, 0, 255).astype(np.uint8)).astype(np.float32) / 255
    darkness = 1 - small
    return s_curve(darkness, np.percentile(darkness, spec.mid), spec.k)


def levels(darkness: np.ndarray, floor: float, top: float = 0.97) -> np.ndarray:
    """Each cell's step: the one whose ink matches its darkness. No dithering, so flat areas
    stay calm."""
    target = np.clip((darkness - floor) / (top - floor), 0, 1) * COVERAGE[-1]
    return np.abs(target[..., None] - COVERAGE).argmin(-1)


def encode(grid: np.ndarray) -> str:
    """Run-length encoding, row by row: a step's letter, then the run's length if above 1."""
    runs = []
    for step, run in groupby(grid.ravel().tolist()):
        count = sum(1 for _ in run)
        runs.append(LETTERS[step] + (str(count) if count > 1 else ""))
    return "".join(runs)


def build(image: Path) -> dict[str, dict]:
    grids = {}
    for name, spec in SPECS.items():
        grid = levels(tone(luminance(image, spec.box), spec), spec.floor)
        grids[name] = {"cols": int(grid.shape[1]), "rows": int(grid.shape[0]), "data": encode(grid)}
    return grids


def write_module(grids: dict[str, dict], path: Path) -> None:
    ramp = [
        {"glyph": glyph, "weight": weight, "reversed": rev, "coverage": float(coverage)}
        for (glyph, weight, rev), coverage in zip(RAMP, COVERAGE, strict=True)
    ]
    lines = [
        "// Generated by `make glyphs` (backend/scripts/glyphs.py) from Charles Holroyd's",
        "// etching Daedalus (1895, public domain). Do not edit by hand.",
        "",
        "export type GlyphStep = {",
        "  glyph: string;",
        "  weight: 400 | 700;",
        "  reversed: boolean;",
        "  coverage: number;",
        "};",
        "export type GlyphGrid = { cols: number; rows: number; data: string };",
        "",
        "// From paper to ink, with the share of its cell each step inks; the last four are",
        "// letters cut out of a solid cell.",
        f"export const RAMP: readonly GlyphStep[] = {json.dumps(ramp, ensure_ascii=False)};",
        "",
        "// Each grid's steps, row by row: a letter a-m per step, then the run's length.",
        "export const GRIDS = {",
        *(
            f'  "{name}": {{ cols: {grid["cols"]}, rows: {grid["rows"]}, data: "{grid["data"]}" }},'
            for name, grid in grids.items()
        ),
        "} satisfies Record<string, GlyphGrid>;",
        "",
        "export type GridName = keyof typeof GRIDS;",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw Holroyd's Daedalus in Greek letters.")
    parser.add_argument("image", type=Path, help="the etching, a 736 x 942 copy")
    parser.add_argument(
        "--out", type=Path, default=OUTPUT, help="where to write the grids (TypeScript)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    image = args.image.expanduser()
    if not image.is_absolute() and not image.exists():
        image = REPO_ROOT / image  # `make glyphs` runs from backend/; paths come from the root
    grids = build(image)
    write_module(grids, args.out)
    for name, grid in grids.items():
        size = f"{grid['cols']} x {grid['rows']} letters"
        print(f"  {name:<16} {size:<20} {len(grid['data']):>6,} characters encoded")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main(parse_args())

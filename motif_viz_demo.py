#!/usr/bin/env python
"""
motif_viz_demo.py

Pipeline:
  1) Load a simplified motif library (motifs_cof_from_library.json).
  2) Select a few "interesting" motifs.
  3) For each motif, construct a toy local-environment CIF:
       - central atom at the cell centre
       - neighbour atoms arranged evenly on a circle
  4) Render each CIF to a PNG using VESTA via VastraVisualise.

Notes:
  - Requires VESTA installed and discoverable (VESTA_EXE env var or on PATH).
  - CIFs here are schematic cartoons of the motif, *not* real COF fragments.
"""

import argparse
import json
import math
from pathlib import Path

from VastraVisualise import VastraVisualise  # your tool class


# ---------------------------------------------------------------------
# 1. Motif selection
# ---------------------------------------------------------------------


def load_motifs(path: Path):
    """Load the simplified motif library (list of {name, central_species, neighbor_species_counts, ...})."""
    with path.open("r") as f:
        motifs = json.load(f)
    if not isinstance(motifs, list):
        raise ValueError("Expected a JSON list of motifs in the library.")
    return motifs


def motif_score(m):
    """
    Heuristic 'interestingness' score:
      - more 'heavy' / less common elements are favoured,
      - more distinct neighbour species,
      - more total neighbours.
    """
    counts = m.get("neighbor_species_counts", {})
    total = sum(counts.values())
    distinct = len(counts)
    heavies = sum(1 for el in counts if el not in {"H", "C", "N", "O"})
    return (heavies, distinct, total)


def pick_interesting_motifs(motifs, k: int = 4):
    """Return the top-k motifs by our heuristic score."""
    sorted_m = sorted(motifs, key=motif_score, reverse=True)
    return sorted_m[:k]


# ---------------------------------------------------------------------
# 2. CIF construction
# ---------------------------------------------------------------------


def build_cif_for_motif(motif: dict, a: float = 10.0) -> str:
    """
    Build a toy CIF with:
      - P1 cell
      - central atom at (0.5, 0.5, 0.5)
      - neighbours arranged evenly around a circle in the xy-plane.

    All positions are fractional coordinates.
    """
    name = motif["name"]
    central = motif["central_species"]
    counts = motif.get("neighbor_species_counts", {})

    sites_lines = []

    # Central atom at the cell centre
    sites_lines.append(f"  {central}0  {central}  0.5  0.5  0.5")

    total_neighbors = sum(counts.values())
    if total_neighbors > 0:
        r = 0.25  # radius in fractional coordinates (keeps atoms safely in the cell)
        idx = 0
        for elem in sorted(counts.keys()):
            for _ in range(counts[elem]):
                theta = 2.0 * math.pi * idx / total_neighbors
                x = 0.5 + r * math.cos(theta)
                y = 0.5 + r * math.sin(theta)
                z = 0.5
                sites_lines.append(
                    f"  {elem}{idx+1}  {elem}  {x:.5f}  {y:.5f}  {z:.5f}"
                )
                idx += 1

    sites_block = "\n".join(sites_lines)

    cif = f"""data_{name}
_symmetry_space_group_name_H-M   'P1'
_cell_length_a   {a}
_cell_length_b   {a}
_cell_length_c   {a}
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 1

loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
{sites_block}
"""
    return cif


# ---------------------------------------------------------------------
# 3. VESTA rendering via VastraVisualise
# ---------------------------------------------------------------------


def render_with_vesta(cif_path: Path, png_path: Path, vesta_exe: str | None = None):
    """
    Use your VastraVisualise tool to export a PNG from a CIF file.
    """
    tool = VastraVisualise(
        vesta_exe=vesta_exe,
        output_png=str(png_path),
        scale=2,
        nogui=True,
    )

    result = tool._run(str(cif_path))
    print(result)


# ---------------------------------------------------------------------
# 4. Main driver
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pick motifs, make toy CIFs, and visualise them with VESTA."
    )
    parser.add_argument(
        "--motif-lib",
        type=Path,
        default=Path("motifs_cof_from_library.json"),
        help="Path to motifs_cof_from_library.json (simplified motif library).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("motif_viz"),
        help="Directory where CIFs and PNGs will be written.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="Number of motifs to visualise.",
    )
    parser.add_argument(
        "--vesta-exe",
        type=str,
        default=None,
        help="Optional: explicit path to VESTA executable. "
             "Otherwise VESTA_EXE or PATH will be used.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load and pick motifs
    motifs = load_motifs(args.motif_lib)
    selected = pick_interesting_motifs(motifs, k=args.num_examples)

    print("Selected motifs:")
    for m in selected:
        counts = m.get("neighbor_species_counts", {})
        print(f"  - {m['name']}: {m['central_species']} with neighbours {counts}")

    # For each motif: write CIF, then render to PNG
    for m in selected:
        name = m["name"]
        cif_text = build_cif_for_motif(m)

        cif_path = args.out_dir / f"{name}.cif"
        png_path = args.out_dir / f"{name}.png"

        cif_path.write_text(cif_text)
        print(f"\nWrote CIF for motif {name} -> {cif_path}")

        try:
            render_with_vesta(cif_path, png_path, vesta_exe=args.vesta_exe)
            print(f"Rendered PNG -> {png_path}")
        except Exception as e:
            print(f"VESTA rendering failed for {name}: {e}")


if __name__ == "__main__":
    main()

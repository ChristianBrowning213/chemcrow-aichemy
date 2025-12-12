#!/usr/bin/env python
"""
motif_viz_shapes.py

- Reads motifs_cof_from_library.json
- Picks motifs with *diverse* geometries (linear, trigonal_planar, tetrahedral, etc.)
- For each motif, builds a toy CIF with that local geometry:
    * central atom at the cell centre
    * neighbours placed according to geometry directions
- Neighbours are placed at ~0.8 * max_distance from the central atom so VESTA
  will normally draw "bond" lines automatically.
- Renders each CIF with VastraVisualise (VESTA wrapper), writing a separate PNG
  per motif.
"""

import os

os.environ["VESTA_EXE"] = r"C:\Users\brown\Documents\VESTA-win64\VESTA.exe"
# ---------------------------------------------------------------------

import argparse
import json
import math
from pathlib import Path

from VastraVisualise import VastraVisualise  # your existing tool


# ---------------------------------------------------------------------
# Motif loading / selection
# ---------------------------------------------------------------------


def load_motifs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        motifs = json.load(f)
    if not isinstance(motifs, list):
        raise ValueError(
            "Expected a JSON list of motifs in motifs_cof_from_library.json"
        )
    return motifs


def motif_coordination(motif: dict) -> int:
    counts = motif.get("neighbor_species_counts", {})
    return int(sum(counts.values()))


def motif_score(motif: dict):
    """
    Heuristic 'interestingness' score:
      - more 'heavy' / less common elements are favoured,
      - more distinct neighbour species,
      - more total neighbours.
    """
    counts = motif.get("neighbor_species_counts", {})
    total = sum(counts.values())
    distinct = len(counts)
    heavies = sum(1 for el in counts if el not in {"H", "C", "N", "O"})
    return (heavies, distinct, total)


def choose_geometry(coord: int) -> str:
    """
    Map coordination number to a geometry name.
    """
    if coord <= 1:
        return "linear"
    if coord == 2:
        return "linear"
    if coord == 3:
        return "trigonal_planar"
    if coord == 4:
        return "tetrahedral"
    if coord == 5:
        return "trigonal_bipyramidal"
    if coord == 6:
        return "octahedral"
    return "circle"


def pick_motifs(motifs, k: int):
    """
    Pick motifs with *diverse geometries*.

    1) Compute coordination for each motif.
    2) Map to geometry via choose_geometry(coord).
    3) Bucket motifs by geometry and sort each bucket by motif_score.
    4) Round-robin across buckets until we have k motifs or run out.
    """
    buckets: dict[str, list[dict]] = {}
    for m in motifs:
        coord = motif_coordination(m)
        geom = choose_geometry(coord)
        buckets.setdefault(geom, []).append(m)

    for geom, bucket in buckets.items():
        bucket.sort(key=motif_score, reverse=True)

    geom_order = [
        "linear",
        "trigonal_planar",
        "tetrahedral",
        "trigonal_bipyramidal",
        "octahedral",
        "circle",
    ]

    chosen: list[dict] = []
    while len(chosen) < k:
        made_progress = False
        for geom in geom_order:
            bucket = buckets.get(geom, [])
            if bucket:
                chosen.append(bucket.pop(0))
                made_progress = True
                if len(chosen) >= k:
                    break
        if not made_progress:
            break

    return chosen


# ---------------------------------------------------------------------
# Geometry templates
# ---------------------------------------------------------------------


def geometry_directions(geom: str, n: int):
    """
    Return up to `n` direction vectors (vx, vy, vz) of unit length
    for the requested geometry.
    """
    if geom == "linear":
        base = [(1, 0, 0), (-1, 0, 0)]

    elif geom == "trigonal_planar":
        base = []
        for i in range(3):
            theta = 2 * math.pi * i / 3
            base.append((math.cos(theta), math.sin(theta), 0.0))

    elif geom == "tetrahedral":
        base = [
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, -1),
            (1, -1, -1),
        ]

    elif geom == "octahedral":
        base = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

    elif geom == "trigonal_bipyramidal":
        base = []
        for i in range(3):
            theta = 2 * math.pi * i / 3
            base.append((math.cos(theta), math.sin(theta), 0.0))
        base.append((0.0, 0.0, 1.0))
        base.append((0.0, 0.0, -1.0))

    else:  # "circle" or fallback
        base = []
        m = max(n, 3)
        for i in range(m):
            theta = 2 * math.pi * i / m
            base.append((math.cos(theta), math.sin(theta), 0.0))

    normed = []
    for vx, vy, vz in base:
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm == 0:
            continue
        normed.append((vx / norm, vy / norm, vz / norm))

    if not normed:
        normed = [(1.0, 0.0, 0.0)]

    out = []
    idx = 0
    while len(out) < n:
        out.append(normed[idx % len(normed)])
        idx += 1

    return out


# ---------------------------------------------------------------------
# CIF construction (with distances tuned for VESTA bonds)
# ---------------------------------------------------------------------


def build_cif_for_motif(motif: dict, geometry: str | None = None) -> str:
    """
    Build a toy CIF for a motif with a specific geometry.

    - Central atom at (0.5, 0.5, 0.5)
    - Neighbours arranged according to 'geometry'
    - Cell length a is tied to motif['max_distance'] so that neighbour
      distance ~= 0.8 * max_distance → VESTA should draw bonds.
    """
    name = motif["name"]
    central = motif["central_species"]
    counts = motif.get("neighbor_species_counts", {})
    max_d = float(motif.get("max_distance", 2.0))
    coord = int(sum(counts.values()))

    if geometry is None or geometry == "auto":
        geometry = choose_geometry(coord)

    a = max(4.0, max_d * 3.0)

    target_dist = 0.8 * max_d
    r_frac = min(0.3, target_dist / a)

    cx = cy = cz = 0.5

    sites_lines = []
    sites_lines.append(f"  {central}0  {central}  {cx:.5f}  {cy:.5f}  {cz:.5f}")

    if coord > 0:
        dirs = geometry_directions(geometry, coord)
        idx = 0
        for elem in sorted(counts.keys()):
            for _ in range(counts[elem]):
                vx, vy, vz = dirs[idx]
                x = cx + r_frac * vx
                y = cy + r_frac * vy
                z = cz + r_frac * vz
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                z = max(0.05, min(0.95, z))
                sites_lines.append(
                    f"  {elem}{idx+1}  {elem}  {x:.5f}  {y:.5f}  {z:.5f}"
                )
                idx += 1

    sites_block = "\n".join(sites_lines)

    cif = f"""data_{name}
_symmetry_space_group_name_H-M   'P1'
_cell_length_a   {a:.5f}
_cell_length_b   {a:.5f}
_cell_length_c   {a:.5f}
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
# VESTA via VastraVisualise
# ---------------------------------------------------------------------


def render_with_vesta(cif_path: Path, png_path: Path):
    """
    Call VastraVisualise but override the output path so each motif
    gets its own PNG.
    """
    png_path = png_path.resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    tool = VastraVisualise(output_png=str(png_path))
    result = tool._run(str(cif_path))
    print(result)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualise motifs with different local geometries using VESTA."
    )
    parser.add_argument(
        "--motif-lib",
        type=Path,
        default=Path("motifs_cof_from_library.json"),
        help="Path to motifs_cof_from_library.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("motif_viz_shapes"),
        help="Output directory for CIFs and PNGs",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of motifs to visualise",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default="auto",
        choices=[
            "auto",
            "linear",
            "trigonal_planar",
            "tetrahedral",
            "trigonal_bipyramidal",
            "octahedral",
            "circle",
        ],
        help="Force a specific geometry, or 'auto' based on coordination",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motifs = load_motifs(args.motif_lib)
    selected = pick_motifs(motifs, k=args.num_examples)

    print("Selected motifs:")
    for m in selected:
        counts = m.get("neighbor_species_counts", {})
        coord = motif_coordination(m)
        geom = args.geometry if args.geometry != "auto" else choose_geometry(coord)
        print(f"  - {m['name']}: coord={coord}, geom={geom}, neighbours={counts}")

    for m in selected:
        coord = motif_coordination(m)
        geom = args.geometry if args.geometry != "auto" else choose_geometry(coord)

        cif_text = build_cif_for_motif(m, geometry=geom)
        cif_path = args.out_dir / f"{m['name']}_{geom}.cif"
        cif_path.write_text(cif_text, encoding="utf-8")
        print(f"\nWrote CIF for motif {m['name']} ({geom}) -> {cif_path}")

        png_path = args.out_dir / f"{m['name']}_{geom}.png"
        try:
            render_with_vesta(cif_path, png_path)
        except Exception as e:
            print(f"VESTA rendering failed for {m['name']} ({geom}): {e}")


if __name__ == "__main__":
    main()

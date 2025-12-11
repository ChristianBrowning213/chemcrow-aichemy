import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _base_name(name: str) -> str:
    """
    Strip trailing '_C<number>' to get the motif 'family' name.

    Examples:
        'BC1O2_C0' -> 'BC1O2'
        'BC1O2_C12' -> 'BC1O2'
        'CC3' -> 'CC3'  (unchanged if no suffix)
    """
    return re.sub(r"_C\d+$", "", name)


def strip_motif_library_family(
    in_path: str | Path,
    out_path: str | Path,
    *,
    max_dist_scale: float = 1.1,
    combine_mode: str = "auto",
    spread_tol: float = 0.10,
) -> None:
    """
    Convert a full motif_library.json into a minimal, MotifPattern-compatible file,
    collapsing cluster variants into 'base families'.

    INPUT schema (current library, abbreviated):

        {
          "motifs": [
            {
              "id": "BC1O2_C0",
              "center_species": "B",
              "composition_counts": {"O": 2, "C": 1},
              "coordination": 3,
              "op_mean": [...],
              "op_std": [...],
              "prototype_offsets": [...],
              "cluster_size": 34,
              "meta": {
                "feature_center": [...],
                ...
              }
            },
            ...
          ]
        }

    FAMILY COLLAPSING:

    - All motifs whose name/id differ only by a '_C<number>' suffix are grouped
      into a single "family" based on the base name, e.g.:

          'BC1O2_C0', 'BC1O2_C1', 'BC1O2_C2' -> family 'BC1O2'

    - We assume within a family:
        * central_species is the same
        * composition_counts is the same
      If not, we raise a ValueError (since that indicates inconsistent data).

    max_distance per cluster:

    - For each cluster motif, we derive a candidate distance:

          base = meta.feature_center[-1]
          max_distance_cluster = base * max_dist_scale

      If feature_center is missing, we fall back to a default (2.0 Å).

    FAMILY max_distance combination:

    - Given distances d_1, ..., d_k for a family, we compute:

        d_mean = average(d_i)
        d_max  = max(d_i)

      combine_mode controls what we use:

        - "max":  use d_max
        - "mean": use d_mean
        - "auto": if d_max <= d_mean * (1 + spread_tol), use d_mean;
                  otherwise use d_max.

      This matches the idea:
        - If clusters are similar, use an average.
        - If one cluster is significantly longer, use the safe maximum.

    OUTPUT schema (MotifPattern-compatible):

        [
          {
            "name": "BC1O2",
            "central_species": "B",
            "neighbor_species_counts": {"O": 2, "C": 1},
            "max_distance": 1.73
          },
          ...
        ]
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    raw = json.loads(in_path.read_text())
    if isinstance(raw, dict) and "motifs" in raw:
        motifs: List[Dict[str, Any]] = raw["motifs"]
    elif isinstance(raw, list):
        motifs = raw
    else:
        raise ValueError(f"Unrecognised motif JSON format in {in_path}")

    # family_key -> dict with:
    #   - central_species: str
    #   - neighbor_species_counts: Dict[str, int]
    #   - distances: List[float]
    families: Dict[str, Dict[str, Any]] = {}

    for m in motifs:
        name = m.get("id") or m.get("name") or "UNKNOWN"
        base = _base_name(name)

        central = m["center_species"]
        comp_counts: Dict[str, int] = m["composition_counts"]

        meta = m.get("meta", {}) or {}
        feature_center = meta.get("feature_center", []) or []

        # Derive distance per cluster
        if feature_center:
            base_dist = float(feature_center[-1])
            max_distance = base_dist * max_dist_scale
        else:
            max_distance = 2.0

        if base not in families:
            families[base] = {
                "central_species": central,
                "neighbor_species_counts": comp_counts,
                "distances": [max_distance],
            }
        else:
            fam = families[base]
            # Sanity check: central species and composition should match within a family
            if fam["central_species"] != central:
                raise ValueError(
                    f"Inconsistent center_species in family {base}: "
                    f"{fam['central_species']} vs {central}"
                )
            if fam["neighbor_species_counts"] != comp_counts:
                raise ValueError(
                    f"Inconsistent composition_counts in family {base}: "
                    f"{fam['neighbor_species_counts']} vs {comp_counts}"
                )
            fam["distances"].append(max_distance)

    stripped: List[Dict[str, Any]] = []

    for base, data in families.items():
        distances: List[float] = data["distances"]
        if not distances:
            # Should not happen, but guard anyway
            chosen = 2.0
        else:
            d_mean = sum(distances) / len(distances)
            d_max = max(distances)

            if combine_mode == "max":
                chosen = d_max
            elif combine_mode == "mean":
                chosen = d_mean
            elif combine_mode == "auto":
                if d_max <= d_mean * (1.0 + spread_tol):
                    chosen = d_mean
                else:
                    chosen = d_max
            else:
                raise ValueError(
                    f"Unknown combine_mode '{combine_mode}'. "
                    "Use 'max', 'mean', or 'auto'."
                )

        stripped.append(
            {
                "name": base,
                "central_species": data["central_species"],
                "neighbor_species_counts": data["neighbor_species_counts"],
                "max_distance": chosen,
            }
        )

    stripped.sort(key=lambda x: x["name"])

    out_path.write_text(json.dumps(stripped, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(stripped)} motif families (from {len(motifs)} clusters) "
        f"to {out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Collapse a full motif_library.json into a minimal MotifPattern JSON, "
            "grouping cluster variants (e.g. 'BC1O2_C0', 'BC1O2_C1') "
            "into base families (e.g. 'BC1O2')."
        )
    )
    parser.add_argument("in_path", help="path to full motif_library.json")
    parser.add_argument(
        "out_path",
        help="output path for minimal motifs JSON (MotifPattern schema)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="safety factor on per-cluster max_distance (default 1.1)",
    )
    parser.add_argument(
        "--combine-mode",
        choices=["auto", "max", "mean"],
        default="auto",
        help=(
            "how to combine cluster distances per family: "
            "'auto' (default), 'max', or 'mean'"
        ),
    )
    parser.add_argument(
        "--spread-tol",
        type=float,
        default=0.10,
        help=(
            "relative spread tolerance for 'auto' mode. "
            "If d_max <= d_mean * (1 + spread_tol), use mean; "
            "otherwise use max. Default 0.10 (10%%)."
        ),
    )

    args = parser.parse_args()

    strip_motif_library_family(
        args.in_path,
        args.out_path,
        max_dist_scale=args.scale,
        combine_mode=args.combine_mode,
        spread_tol=args.spread_tol,
    )

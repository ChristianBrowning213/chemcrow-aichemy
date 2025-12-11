import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain.tools import BaseTool
from pymatgen.core import Structure
from pymatgen.analysis.local_env import MinimumDistanceNN

# Optional: handle NumPy types when they appear
try:
    import numpy as np
except ImportError:
    np = None

MOTIF_ERROR_PREFIX = "[MOTIF_TOOL_ERROR]"


def _motif_fail(msg: str) -> str:
    """
    Return a compact, structured error string that the agent can detect.
    """
    # Keep the message short so it does not bloat context.
    return f"{MOTIF_ERROR_PREFIX} {msg}"


def _json_default(o):
    """
    Helper for json.dumps to convert NumPy types to plain Python types.
    """
    if np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    # Fallback: just stringify anything unknown
    return str(o)


# =========================
# Motif representation
# =========================

@dataclass
class MotifPattern:
    """
    Simple chemical motif: central species + neighbor species counts + max distance.

    Example JSON entry:
    {
      "name": "TiO6_oct",
      "central_species": "Ti",
      "neighbor_species_counts": {"O": 6},
      "max_distance": 2.3
    }
    """
    name: str
    central_species: str
    neighbor_species_counts: Dict[str, int]
    max_distance: float
    extra: Dict[str, Any] = field(default_factory=dict)

def _load_motif_library_from_file(path: str | Path) -> List[MotifPattern]:
    """
    Load motifs from a *simple* JSON file that is already in MotifPattern schema:

        [
          {
            "name": "C_sp2_like",
            "central_species": "C",
            "neighbor_species_counts": {"C": 2, "N": 1},
            "max_distance": 1.7
          },
          ...
        ]

    This assumes you've preprocessed any complex libraries into this format.
    """
    data = json.loads(Path(path).read_text())

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of motifs in {path!s}, "
            "each with name / central_species / neighbor_species_counts / max_distance."
        )

    patterns: List[MotifPattern] = []
    for entry in data:
        patterns.append(
            MotifPattern(
                name=entry["name"],
                central_species=entry["central_species"],
                neighbor_species_counts=entry["neighbor_species_counts"],
                max_distance=float(entry["max_distance"]),
                # Optional: keep any extra keys if present
                extra={
                    k: v
                    for k, v in entry.items()
                    if k
                    not in {
                        "name",
                        "central_species",
                        "neighbor_species_counts",
                        "max_distance",
                    }
                },
            )
        )
    return patterns



def _load_motif_library(
    motifs: Optional[Sequence[Dict]] = None,
    motif_library_path: Optional[str] = None,
) -> List[MotifPattern]:
    """
    Load motif patterns either from an inline list (motifs)
    or from a JSON file (motif_library_path).
    """
    if motifs is not None:
        return [MotifPattern(**m) for m in motifs]
    if motif_library_path is not None:
        return _load_motif_library_from_file(motif_library_path)
    raise ValueError("No motif definitions provided: supply 'motifs' or 'motif_library_path'.")


# =========================
# Core motif logic
# =========================

def _get_neighbor_info(
    structure: Structure,
    central_index: int,
    max_distance: float,
    nn_strategy=None,
):
    if nn_strategy is None:
        nn_strategy = MinimumDistanceNN()
    neighs = nn_strategy.get_nn_info(structure, central_index)

    neighbors = []
    central_site = structure[central_index]
    for info in neighs:
        j = info["site_index"]
        dist = central_site.distance(structure[j])
        if dist <= max_distance:
            species_j = (
                structure[j].specie.symbol
                if hasattr(structure[j], "specie")
                else structure[j].species_string
            )
            neighbors.append((j, species_j, dist))

    return neighbors


def _match_pattern(
    structure: Structure,
    central_index: int,
    pattern: MotifPattern,
) -> Optional[Dict]:
    # Check central species first
    central_site = structure[central_index]
    central_species = (
        central_site.specie.symbol
        if hasattr(central_site, "specie")
        else central_site.species_string
    )
    if central_species != pattern.central_species:
        return None

    neighbors = _get_neighbor_info(structure, central_index, pattern.max_distance)
    if not neighbors:
        return None

    # Count neighbors by species
    counts: Dict[str, int] = {}
    for _, species, _ in neighbors:
        counts[species] = counts.get(species, 0) + 1

    # Fail fast if we don't have enough of any required species
    for sp, req in pattern.neighbor_species_counts.items():
        if counts.get(sp, 0) < req:
            return None

    # Choose closest neighbors for each species
    chosen_neighbors: List[int] = []
    for sp, req in pattern.neighbor_species_counts.items():
        cand = [(idx, dist) for (idx, species, dist) in neighbors if species == sp]
        cand.sort(key=lambda x: x[1])
        if len(cand) < req:
            return None
        chosen_neighbors.extend(idx for idx, _ in cand[:req])

    return {
        "motif_name": pattern.name,
        "central_index": central_index,
        "neighbor_indices": sorted(set(chosen_neighbors)),
    }


def find_motif_occurrences(
    structure: Structure,
    pattern: MotifPattern,
) -> List[Dict]:
    """
    Find all occurrences of `pattern` in `structure`.
    """
    matches: List[Dict] = []
    for i in range(len(structure)):
        m = _match_pattern(structure, i, pattern)
        if m is not None:
            matches.append(m)
    return matches


def decompose_structure(
    structure: Structure,
    motif_library: Sequence[MotifPattern],
    allowed_motifs: Optional[Sequence[str]] = None,
    allow_overlap: bool = True,
) -> Dict:
    """
    Decompose a structure into motif instances from `motif_library`.
    """
    if allowed_motifs is not None:
        allowed = set(allowed_motifs)
        motifs = [m for m in motif_library if m.name in allowed]
    else:
        motifs = list(motif_library)

    raw_matches: List[Dict] = []
    for pattern in motifs:
        occs = find_motif_occurrences(structure, pattern)
        raw_matches.extend(occs)

    if allow_overlap:
        assigned = raw_matches
    else:
        # Greedy: prefer motifs with largest number of sites
        raw_matches.sort(key=lambda m: len(m["neighbor_indices"]) + 1, reverse=True)
        used_sites = set()
        assigned = []
        for m in raw_matches:
            sites = {m["central_index"], *m["neighbor_indices"]}
            if sites & used_sites:
                continue
            assigned.append(m)
            used_sites |= sites

    all_sites = set(range(len(structure)))
    covered_sites = set()
    for m in assigned:
        covered_sites.add(m["central_index"])
        covered_sites.update(m["neighbor_indices"])

    unassigned_sites = sorted(all_sites - covered_sites)

    return {
        "motifs": assigned,
        "unassigned_sites": unassigned_sites,
    }


# =========================
# Structure comparison logic
# =========================

def compare_two_structures(
    structure1: Structure,
    structure2: Structure,
    motif_library: Sequence[MotifPattern],
    allowed_motifs: Optional[Sequence[str]] = None,
    allow_overlap: bool = True,
) -> Dict:
    """
    Decompose both structures and compare motifs.
    """
    decomp1 = decompose_structure(
        structure1, motif_library, allowed_motifs=allowed_motifs, allow_overlap=allow_overlap
    )
    decomp2 = decompose_structure(
        structure2, motif_library, allowed_motifs=allowed_motifs, allow_overlap=allow_overlap
    )

    index1: Dict[str, List[Dict]] = defaultdict(list)
    index2: Dict[str, List[Dict]] = defaultdict(list)

    for m in decomp1["motifs"]:
        index1[m["motif_name"]].append(m)
    for m in decomp2["motifs"]:
        index2[m["motif_name"]].append(m)

    names1 = set(index1.keys())
    names2 = set(index2.keys())

    shared = []
    for name in sorted(names1 & names2):
        shared.append(
            {
                "motif_name": name,
                "count_1": len(index1[name]),
                "count_2": len(index2[name]),
                "instances_1": index1[name],
                "instances_2": index2[name],
            }
        )

    result = {
        "decomp_1": decomp1,
        "decomp_2": decomp2,
        "shared_motifs": shared,
        "unique_to_1": sorted(names1 - names2),
        "unique_to_2": sorted(names2 - names1),
    }
    return result


# =========================
# Small summaries for the LLM
# =========================

def _summarise_decomposition_for_llm(result: Dict, max_examples_per_motif: int = 3) -> Dict:
    """
    Compress a full decomposition result into something LLM-friendly:
    - counts per motif_name
    - total motif instances
    - number of unassigned sites
    - a few example instances per motif
    """
    motifs: List[Dict] = result.get("motifs", [])
    unassigned_sites = result.get("unassigned_sites", [])

    counts: Dict[str, int] = defaultdict(int)
    examples: Dict[str, List[Dict]] = defaultdict(list)
    for m in motifs:
        name = m.get("motif_name", "UNKNOWN")
        counts[name] += 1
        if len(examples[name]) < max_examples_per_motif:
            examples[name].append(
                {
                    "central_index": int(m.get("central_index", -1)),
                    "neighbor_indices": m.get("neighbor_indices", []),
                }
            )

    return {
        "total_motif_instances": len(motifs),
        "motif_counts": dict(counts),
        "unassigned_sites_count": len(unassigned_sites),
        "unassigned_sites_sample": unassigned_sites[:20],
        "example_instances": {k: v for k, v in examples.items()},
    }


def _summarise_comparison_for_llm(result: Dict, max_examples_per_motif: int = 3) -> Dict:
    """
    Compress the full comparison result:
    - summaries of decomp_1 and decomp_2
    - shared motif names + counts
    - unique motif names
    """
    decomp1 = result.get("decomp_1", {})
    decomp2 = result.get("decomp_2", {})
    shared = result.get("shared_motifs", [])
    unique1 = result.get("unique_to_1", [])
    unique2 = result.get("unique_to_2", [])

    summary_decomp1 = _summarise_decomposition_for_llm(
        {"motifs": decomp1.get("motifs", []), "unassigned_sites": decomp1.get("unassigned_sites", [])},
        max_examples_per_motif=max_examples_per_motif,
    )
    summary_decomp2 = _summarise_decomposition_for_llm(
        {"motifs": decomp2.get("motifs", []), "unassigned_sites": decomp2.get("unassigned_sites", [])},
        max_examples_per_motif=max_examples_per_motif,
    )

    shared_summary = []
    for s in shared:
        shared_summary.append(
            {
                "motif_name": s.get("motif_name", "UNKNOWN"),
                "count_1": int(s.get("count_1", 0)),
                "count_2": int(s.get("count_2", 0)),
            }
        )

    return {
        "decomp_1_summary": summary_decomp1,
        "decomp_2_summary": summary_decomp2,
        "shared_motifs": shared_summary,
        "unique_to_1": unique1,
        "unique_to_2": unique2,
    }


# =========================
# ChemCrow tools
# =========================

class MotifDecompositionTool(BaseTool):
    """
    ChemCrow tool for motif-level analysis of crystal structures in CIF format.

    This tool takes a crystal structure (CIF) and a library of predefined
    local motifs (e.g. coordination polyhedra or fragment patterns), and
    decomposes the structure into motif instances.

    Under the hood:
      - It uses pymatgen's MinimumDistanceNN to find neighbours for each site.
      - Each motif pattern is defined by a central species, a required count
        of neighbour species, and a maximum radial cutoff.
      - For each site, the tool checks whether the local environment matches
        any motif in the library and records all matches (optionally allowing
        overlaps).

    Why this is useful:
      - It lets you move from raw atomic coordinates to a "motif vocabulary"
        of the structure (e.g. how many BO3 vs BO4 vs linkers, etc.).
      - This motif fingerprint can be used to compare structures, build motif
        statistics over a dataset, or feed downstream models (e.g. ML or
        retrosynthesis logic).
      - It provides both a compact summary (counts, examples) and a full
        JSON dump of all motif instances and unassigned sites for post-hoc
        analysis outside the LLM.

    The quality of the decomposition depends on the motif library you supply:
    max_distance and neighbour species counts must be tuned to your chemistry.
    """

    name: str = "MotifDecomposition"
    description: str = (
        "Decompose a CIF crystal structure into local coordination motifs using "
        "a predefined motif library. The input MUST be a JSON string with: "
        "'mode' ('search' | 'all' | 'from-list'), 'cif_path', and either "
        "'motifs' (inline motif definitions) or 'motif_library_path' (JSON file). "
        "The tool uses MinimumDistanceNN to detect neighbours and matches each site "
        "to motifs defined by central species, neighbour species counts, and a "
        "max distance cutoff. "
        "Use this when you want a motif-level view of a structure: counts of each "
        "motif type, example instances, and which sites are not covered by any motif. "
        "It returns a compact JSON summary for the agent and writes the full result "
        "to 'motif_results/<cif_stem>_motifs_<mode>.json' for detailed offline analysis."
    )
    default_motif_library_path: Optional[str] = None

    def __init__(self, default_motif_library_path: Optional[str] = None):
        super().__init__()
        self.default_motif_library_path = default_motif_library_path
    def _run(self, query: str) -> str:
        try:
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                return _motif_fail(
                    "Invalid input. Expected a JSON string with keys like "
                    "'mode', 'cif_path', and either 'motifs' or 'motif_library_path'."
                )

            mode = params.get("mode", "all")
            cif_path = params.get("cif_path", None)
            if cif_path is None:
                return _motif_fail("Missing required key 'cif_path' in input JSON.")

            motif_defs = params.get("motifs", None)
            motif_library_path = params.get(
                "motif_library_path",
                self.default_motif_library_path,
            )
            allow_overlap = params.get("allow_overlap", True)

            try:
                motif_library = _load_motif_library(
                    motifs=motif_defs,
                    motif_library_path=motif_library_path,
                )
            except Exception as e:
                return _motif_fail(
                    f"Error loading motif library (check 'motifs' or "
                    f"'motif_library_path'): {type(e).__name__}"
                )

            try:
                structure = Structure.from_file(cif_path)
            except Exception as e:
                return _motif_fail(
                    f"Error reading CIF file '{cif_path}': {type(e).__name__}"
                )

            # Build full internal result
            if mode == "search":
                motif_name = params.get("motif_name", None)
                if not motif_name:
                    return _motif_fail(
                        "In 'search' mode you must provide 'motif_name'."
                    )
                try:
                    pattern = next(m for m in motif_library if m.name == motif_name)
                except StopIteration:
                    return _motif_fail(
                        f"Motif '{motif_name}' not found in motif library."
                    )
                occs = find_motif_occurrences(structure, pattern)
                full_result = {
                    "mode": "search",
                    "motif_name": motif_name,
                    "motifs": occs,
                    "unassigned_sites": [],
                }

            elif mode == "from-list":
                allowed = params.get("allowed_motifs", None)
                if not allowed:
                    return _motif_fail(
                        "In 'from-list' mode you must provide 'allowed_motifs'."
                    )
                decomp = decompose_structure(
                    structure,
                    motif_library,
                    allowed_motifs=allowed,
                    allow_overlap=allow_overlap,
                )
                decomp["mode"] = "from-list"
                full_result = decomp

            elif mode == "all":
                decomp = decompose_structure(
                    structure,
                    motif_library,
                    allowed_motifs=None,
                    allow_overlap=allow_overlap,
                )
                decomp["mode"] = "all"
                full_result = decomp

            else:
                return _motif_fail(
                    "Invalid 'mode'. Expected one of: 'search', 'all', 'from-list'."
                )

            # Save full result to disk
            out_dir = Path("motif_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            base = Path(cif_path).stem
            out_path = out_dir / f"{base}_motifs_{mode}.json"
            out_path.write_text(
                json.dumps(full_result, indent=2, default=_json_default),
                encoding="utf-8",
            )

            # Build small summary for the LLM
            summary_core = _summarise_decomposition_for_llm(full_result)
            summary = {
                "mode": mode,
                "cif_path": cif_path,
                "full_result_path": str(out_path),
                **summary_core,
            }

            return json.dumps(summary, indent=2, default=_json_default)

        except Exception as e:
            # Absolute last-resort guard: no stack trace, just a short tag + type
            return _motif_fail(
                f"Unexpected tool-level error in MotifDecomposition: "
                f"{type(e).__name__}"
            )



class MotifComparisonTool(BaseTool):
    """
    Compare motif fingerprints between two crystal structures (two CIF files)
    using a shared motif library.

    For each structure:
      - Runs the same motif decomposition pipeline as MotifDecompositionTool.
      - Builds an index of motif_name -> list of motif instances.

    Then:
      - Identifies motifs present in both structures and reports their counts.
      - Lists motifs that are unique to structure 1 or structure 2.
      - Saves a detailed JSON containing the full decompositions and per-motif
        instance data.

    Why this is useful:
      - It gives a chemically meaningful comparison that goes beyond simple
        formula or cell parameters: you can see how two COFs/MOFs differ in
        local building blocks.
      - Helpful for "is this COF basically the same motif-wise as that one?",
        clustering, or analysing structure–property differences in terms of
        local environments rather than only global descriptors.
    """

    name: str = "MotifComparison"
    description: str = (
        "Compare the motif content of two CIF structures using a shared motif "
        "library. Input MUST be a JSON string with 'cif_path_1', 'cif_path_2', "
        "and either 'motifs' or 'motif_library_path'. Optional keys: "
        "'allowed_motifs' to restrict to a subset and 'allow_overlap' to control "
        "whether motifs can share sites. "
        "The tool decomposes both structures into motif instances, then reports: "
        "(i) a summary of motif counts in each structure, "
        "(ii) motifs shared by both (with counts), and "
        "(iii) motif types unique to each. "
        "Use this when you want to answer questions like 'how do these two COFs "
        "differ in terms of local building blocks?' rather than just comparing "
        "formulae or bulk descriptors. The full comparison is saved to "
        "'motif_results/compare_<name1>_vs_<name2>.json' for deeper inspection."
    )
    
    default_motif_library_path: Optional[str] = None

    def __init__(self, default_motif_library_path: Optional[str] = None):
        super().__init__()
        self.default_motif_library_path = default_motif_library_path

    def _run(self, query: str) -> str:
        try:
            try:
                params = json.loads(query)
            except json.JSONDecodeError:
                return _motif_fail(
                    "Invalid input. Expected a JSON string with keys like "
                    "'cif_path_1', 'cif_path_2', and either 'motifs' or "
                    "'motif_library_path'."
                )

            cif_path_1 = params.get("cif_path_1", None)
            cif_path_2 = params.get("cif_path_2", None)
            if not cif_path_1 or not cif_path_2:
                return _motif_fail(
                    "Both 'cif_path_1' and 'cif_path_2' are required."
                )

            motif_defs = params.get("motifs", None)
            motif_library_path = params.get(
                "motif_library_path",
                self.default_motif_library_path,
            )
            allowed_motifs = params.get("allowed_motifs", None)
            allow_overlap = params.get("allow_overlap", True)

            try:
                motif_library = _load_motif_library(
                    motifs=motif_defs,
                    motif_library_path=motif_library_path,
                )
            except Exception as e:
                return _motif_fail(
                    f"Error loading motif library (check 'motifs' or "
                    f"'motif_library_path'): {type(e).__name__}"
                )

            try:
                struct1 = Structure.from_file(cif_path_1)
            except Exception as e:
                return _motif_fail(
                    f"Error reading CIF file 1 '{cif_path_1}': {type(e).__name__}"
                )

            try:
                struct2 = Structure.from_file(cif_path_2)
            except Exception as e:
                return _motif_fail(
                    f"Error reading CIF file 2 '{cif_path_2}': {type(e).__name__}"
                )

            try:
                full_result = compare_two_structures(
                    struct1,
                    struct2,
                    motif_library,
                    allowed_motifs=allowed_motifs,
                    allow_overlap=allow_overlap,
                )
            except Exception as e:
                return _motif_fail(
                    f"Error during motif comparison: {type(e).__name__}"
                )

            # Save full comparison result to disk
            out_dir = Path("motif_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            base1 = Path(cif_path_1).stem
            base2 = Path(cif_path_2).stem
            out_path = out_dir / f"compare_{base1}_vs_{base2}.json"
            out_path.write_text(
                json.dumps(full_result, indent=2, default=_json_default),
                encoding="utf-8",
            )

            # Small summary for LLM
            summary_core = _summarise_comparison_for_llm(full_result)
            summary = {
                "cif_path_1": cif_path_1,
                "cif_path_2": cif_path_2,
                "full_result_path": str(out_path),
                **summary_core,
            }

            return json.dumps(summary, indent=2, default=_json_default)

        except Exception as e:
            return _motif_fail(
                f"Unexpected tool-level error in MotifComparison: "
                f"{type(e).__name__}"
            )

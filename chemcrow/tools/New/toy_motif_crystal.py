# chemcrow/tools/New/toy_motif_crystal.py

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
#  VESTA helpers
# ----------------------------------------------------------------------

def _resolve_vesta(exe_hint: Optional[str] = None) -> str:
    """
    Resolve the VESTA executable path.

    Priority:
      1. explicit exe_hint argument (if given and exists)
      2. VESTA_EXE environment variable (if set and exists)
      3. shutil.which("VESTA") / shutil.which("VESTA.exe")
      4. a common default Windows installation path (best-effort)
    """
    from shutil import which

    # 1) direct hint
    if exe_hint:
        p = Path(exe_hint)
        if p.exists():
            return str(p.resolve())

    # 2) env var
    env_path = os.getenv("VESTA_EXE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())

    # 3) PATH lookup
    for candidate in ("VESTA", "VESTA.exe"):
        found = which(candidate)
        if found:
            return found

    # 4) very rough default (adjust if needed)
    default_win = Path(r"C:\Program Files\VESTA\VESTA.exe")
    if default_win.exists():
        return str(default_win.resolve())

    raise RuntimeError(
        "Could not resolve VESTA executable. "
        "Set VESTA_EXE env var or pass vesta_exe_hint."
    )


def _launch_vesta(cif_path: Path, png_path: Optional[Path], exe_hint: Optional[str]) -> None:
    """
    Launch VESTA on the given CIF.

    Notes:
      - This currently just opens VESTA with the CIF. If you want automatic PNG
        export, configure VESTA with a macro that exports PNG on startup, or
        adapt the command-line arguments according to your local VESTA setup.
      - We *reserve* a PNG path so each run uses a distinct filename and you
        can configure VESTA to write there.
    """
    exe = _resolve_vesta(exe_hint)

    # Basic command: just open the CIF.
    # If you later want automatic export, you can extend this to include
    # VESTA's script/macro-related CLI flags.
    cmd = [exe, str(cif_path)]

    # Optionally, you can pass the PNG path via env to a macro, etc.
    if png_path is not None:
        os.environ["VESTA_EXPORT_PNG"] = str(png_path)

    subprocess.Popen(cmd)  # non-blocking, let VESTA open in the background


# ----------------------------------------------------------------------
#  CIF builder
# ----------------------------------------------------------------------

def _write_toy_cif(motif_ids: List[str], cif_path: Path) -> None:
    """
    Build a very simple toy CIF that places each motif on a 2×2×1
    simple-cubic grid. This is *not* chemically realistic; it is a
    visual toy to see "motif placements" as pseudo-atoms.

    We encode the motif ID in the atom label; the actual element is
    set to 'C' for everything.
    """
    cif_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple P1 cell
    a = b = c = 10.0
    alpha = beta = gamma = 90.0

    # Up to 4 positions in a 2x2x1 grid
    base_positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
    ]
    if not motif_ids:
        motif_ids = ["DUMMY_MOTIF"]

    # Repeat motifs if fewer than positions, or truncate if more
    n_sites = min(len(base_positions), len(motif_ids))
    used_motifs = motif_ids[:n_sites]

    # Build CIF lines
    lines = []
    lines.append("data_toy_motif_crystal")
    lines.append("_symmetry_space_group_name_H-M    'P 1'")
    lines.append(f"_cell_length_a    {a:.4f}")
    lines.append(f"_cell_length_b    {b:.4f}")
    lines.append(f"_cell_length_c    {c:.4f}")
    lines.append(f"_cell_angle_alpha {alpha:.4f}")
    lines.append(f"_cell_angle_beta  {beta:.4f}")
    lines.append(f"_cell_angle_gamma {gamma:.4f}")
    lines.append("")
    lines.append("loop_")
    lines.append("  _symmetry_equiv_pos_as_xyz")
    lines.append("  'x,y,z'")
    lines.append("")
    lines.append("loop_")
    lines.append("  _atom_site_label")
    lines.append("  _atom_site_type_symbol")
    lines.append("  _atom_site_fract_x")
    lines.append("  _atom_site_fract_y")
    lines.append("  _atom_site_fract_z")
    lines.append("  _atom_site_occupancy")

    for i, (motif_id, (fx, fy, fz)) in enumerate(zip(used_motifs, base_positions)):
        label = f"{i+1}_{motif_id}"
        # We use a dummy element type "C". The motif ID sits in the label.
        lines.append(
            f"  {label}   C   {fx:.5f}   {fy:.5f}   {fz:.5f}   1.0"
        )

    cif_text = "\n".join(lines) + "\n"
    cif_path.write_text(cif_text, encoding="utf-8")


# ----------------------------------------------------------------------
#  Tool definition
# ----------------------------------------------------------------------

class BuildToyMotifCrystalInput(BaseModel):
    motif_ids: List[str] = Field(
        ...,
        description=(
            "List of motif identifiers (e.g. ['OB1P1', 'CB1C2', 'CC1S2']). "
            "Only the first 4 are used in this toy example."
        )
    )
    output_basename: str = Field(
        "toy_motif_crystal",
        description="Base filename (without extension) for CIF/PNG outputs."
    )
    output_dir: str = Field(
        "motif_toy_crystals",
        description="Directory where CIF and PNG files will be written."
    )
    call_vesta: bool = Field(
        True,
        description="If true, launch VESTA/VESTA.exe on the generated CIF."
    )
    vesta_exe_hint: Optional[str] = Field(
        None,
        description="Optional explicit path to the VESTA executable."
    )


class BuildToyMotifCrystal(BaseTool):
    """
    Construct a toy crystal from a small set of motifs, write it as CIF,
    and (optionally) open it in VESTA for visual inspection.

    This is deliberately *didactic* rather than physically realistic: it
    simply places each motif on a simple 2×2×1 grid and encodes the motif
    ID in the atom label so you can see which motif corresponds to which
    pseudo-atom in VESTA.
    """

    name = "BuildToyMotifCrystal"
    description = (
        "Build a toy periodic crystal from a small set of motif IDs, write a CIF "
        "file, and optionally launch VESTA/VESTA.exe to visualise it. Intended "
        "for quick motif-level visualisation, not for physical simulation."
    )

    args_schema = BuildToyMotifCrystalInput

    def _run(
        self,
        motif_ids: List[str],
        output_basename: str = "toy_motif_crystal",
        output_dir: str = "motif_toy_crystals",
        call_vesta: bool = True,
        vesta_exe_hint: Optional[str] = None,
        **kwargs,
    ) -> str:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        cif_path = outdir / f"{output_basename}.cif"
        png_path = outdir / f"{output_basename}.png"

        _write_toy_cif(motif_ids, cif_path)

        vesta_status = "not called"
        if call_vesta:
            try:
                _launch_vesta(cif_path, png_path, vesta_exe_hint)
                vesta_status = f"VESTA launched on {cif_path}"
            except Exception as exc:
                vesta_status = f"FAILED to launch VESTA: {exc!r}"

        return (
            f"Toy motif crystal CIF written to: {cif_path}\n"
            f"Reserved PNG output path:       {png_path}\n"
            f"VESTA status: {vesta_status}"
        )

    async def _arun(self, *args, **kwargs) -> str:
        # Async not needed for now.
        raise NotImplementedError("BuildToyMotifCrystal does not support async")

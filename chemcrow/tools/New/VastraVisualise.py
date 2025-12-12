import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import time
import json
import math

from langchain.tools import BaseTool


# -------------------- VESTA helpers -------------------- #

def _resolve_vesta(exe_hint: Optional[str] = None) -> str:
    """
    Resolve the VESTA executable path.

    Priority:
      1. explicit exe_hint argument (if given and exists)
      2. VESTA_EXE environment variable (if set and exists)
      3. shutil.which("VESTA") / shutil.which("VESTA.exe")
      4. (optional) a common default installation path on Windows
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
    for name in ("VESTA", "VESTA.exe"):
        p = which(name)
        if p:
            return str(Path(p).resolve())

    # 4) optional hard-coded default (edit if you like)
    default_win = Path(r"C:\Program Files\VESTA\VESTA.exe")
    if default_win.exists():
        return str(default_win.resolve())

    raise RuntimeError(
        "VESTA executable not found. "
        "Pass vesta_exe=..., set VESTA_EXE, or add VESTA to PATH."
    )


def _norm(p: Path) -> str:
    """
    VESTA CLI on Windows prefers absolute paths with forward slashes.
    """
    return str(Path(p).resolve()).replace("\\", "/")


def _write_temp_cif_from_text(cif_text: str, tmp_dir: Path) -> Path:
    """
    Write raw CIF text to a temporary file in tmp_dir, return its path.
    """
    cif_path = tmp_dir / "vastra_input.cif"
    cif_path.write_text(cif_text)
    return cif_path


def _run_vesta_export(
    cif_path: Path,
    out_png: Path,
    vesta_exe: Optional[str] = None,
    scale: Optional[int] = 2,
    nogui: bool = True,  # kept for signature compatibility, but ignored
) -> str:
    """
    Call VESTA to export a PNG from a CIF.

    We treat the run as SUCCESS if the PNG file is actually created,
    even if VESTA returns a non-zero code. Only if no PNG appears do we
    raise an error.
    """
    exe = _resolve_vesta(vesta_exe)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-open", _norm(cif_path), "-export_img"]
    if scale is not None:
        cmd += [f"scale={scale}", _norm(out_png)]
    else:
        cmd += [_norm(out_png)]

    creationflags = 0
    if os.name == "nt":
        # Hide console window on Windows
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )

    # Wait briefly for the PNG to appear
    wait_seconds = 5.0
    poll_interval = 0.25
    deadline = time.time() + wait_seconds

    while not out_png.exists() and time.time() < deadline:
        time.sleep(poll_interval)

    if out_png.exists():
        return str(out_png.resolve())

    raise RuntimeError(
        "VESTA command failed (no PNG created).\n"
        f"Return code: {result.returncode}\n"
        f"CMD:\n{' '.join(cmd)}\n\n"
        f"STDOUT:\n{result.stdout or '(empty)'}\n\n"
        f"STDERR:\n{result.stderr or '(empty)'}"
    )


# -------------------- Motif → CIF helpers -------------------- #

def _choose_geometry(coord: int) -> str:
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


def _geometry_directions(geom: str, n: int):
    """
    Return up to `n` direction vectors (vx, vy, vz) of unit length
    for the requested idealised coordination geometry.
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


def _build_cif_for_motif(
    motif: dict,
    geometry: Optional[str] = None,
) -> str:
    """
    Build a toy CIF for an *isolated motif* in MotifPattern schema:

        {
          "name": "TiO6_oct",
          "central_species": "Ti",
          "neighbor_species_counts": {"O": 6},
          "max_distance": 2.3
        }

    - Central atom at (0.5, 0.5, 0.5)
    - Neighbours arranged according to 'geometry'
    - Cell length 'a' tied to max_distance so neighbour distance
      ~ 0.8 * max_distance to encourage VESTA to draw bonds.
    """
    name = motif["name"]
    central = motif["central_species"]
    counts = motif.get("neighbor_species_counts", {})
    max_d = float(motif.get("max_distance", 2.0))
    coord = int(sum(counts.values()))

    if geometry is None or geometry == "auto":
        geometry = _choose_geometry(coord)

    # Simple cubic box big enough that neighbours are not too close to cell edge
    a = max(4.0, max_d * 3.0)

    # Place neighbours at 0.8 * max_d in idealised directions
    target_dist = 0.8 * max_d
    r_frac = min(0.3, target_dist / a)

    cx = cy = cz = 0.5

    sites_lines = []
    sites_lines.append(f"  {central}0  {central}  {cx:.5f}  {cy:.5f}  {cz:.5f}")

    coord = sum(counts.values())
    if coord > 0:
        dirs = _geometry_directions(geometry, coord)
        idx = 0
        for elem in sorted(counts.keys()):
            for _ in range(counts[elem]):
                vx, vy, vz = dirs[idx]
                x = cx + r_frac * vx
                y = cy + r_frac * vy
                z = cz + r_frac * vz
                # Clamp into (0,1) so VESTA doesn't drop anything on boundaries
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


def _load_motif_from_library(
    lib_path: Path,
    motif_name: str,
) -> dict:
    """
    Load a single motif definition from a simple MotifPattern-style JSON
    library (list of dicts).
    """
    data = json.loads(lib_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON *list* of motifs in {lib_path!s} "
            "(MotifPattern schema)."
        )
    for entry in data:
        if entry.get("name") == motif_name:
            return entry
    raise KeyError(f"Motif '{motif_name}' not found in {lib_path!s}")


# ----------------------- ChemCrow / LangChain tool ----------------------- #

class VastraVisualise(BaseTool):
    """
    ChemCrow tool: call VESTA from the command line to render either:

      1. A full crystal structure from CIF (path or raw CIF text), OR
      2. An *isolated motif* defined in the same MotifPattern schema used
         by the motif tools.

    This turns VESTA into a programmatic visualisation backend for both
    crystal files and abstract motif definitions.

    Input modes
    -----------

    (A) CIF mode – legacy (string input):

        - Input: a plain string.
        - If the string is a path to an existing `.cif` file, that file is
          passed directly to VESTA.
        - Otherwise, the string is interpreted as raw CIF text, written to
          a temporary file, then rendered.
        - Uses the instance's `output_png` attribute as the PNG path.

    (B) CIF mode – JSON (allows per-call output filename):

        - Input: a JSON string with either:
            { "mode": "cif", "cif_path": "path/to/file.cif",
              "output_png": "viz/my_cof.png" }
          or
            { "mode": "cif", "cif_text": "<raw CIF text>",
              "output_png": "viz/my_cof.png" }

        - If `output_png` is omitted, falls back to the instance's
          `output_png` attribute.

    (C) Motif mode – JSON (interoperable with motif tools):

        - Input: a JSON string with at least `"mode": "motif"` and either:

            {
              "mode": "motif",
              "motif": { ...MotifPattern dict... },
              "geometry": "auto",
              "output_png": "viz/my_motif.png"
            }

          OR

            {
              "mode": "motif",
              "motif_library_path": "chemcrow/tools/New/motifs_cof_simple.json",
              "motif_name": "BC1O2_C0",
              "geometry": "auto",
              "output_png": "viz/BC1O2_C0.png"
            }

        - The motif dictionary must follow the MotifPattern schema:
          `name`, `central_species`, `neighbor_species_counts`, `max_distance`.

        - `output_png` is optional; if omitted, the instance's `output_png`
          value is used.

    Why this is useful
    ------------------
    - It gives the agent a quick way to produce human-viewable pictures of:
        • full COF/MOF structures (pore geometry, connectivity, CIF sanity),
        • isolated coordination motifs from your motif library or decomposition.
    - Because `output_png` can be specified per call, the agent can render
      multiple COFs or motifs in one run without overwriting images.
    """

    name = "VastraVisualise"
    description = (
        "Render either (a) a full crystal structure from a CIF, or (b) an isolated "
        "motif defined in MotifPattern schema, into a PNG using VESTA.\n\n"
        "CIF usage (string): call with a plain string. If it is a path to an existing "
        "'.cif' file, that file is rendered directly. Otherwise, the string is treated "
        "as raw CIF text. Uses the tool's default output_png path.\n\n"
        "CIF usage (JSON, per-call filenames): call with a JSON string such as\n"
        "  {\"mode\": \"cif\", \"cif_path\": \"C:/.../07000N2_ddec.cif\", "
        "\"output_png\": \"viz/07000N2.png\"}\n"
        "or\n"
        "  {\"mode\": \"cif\", \"cif_text\": \"<full CIF text>\", "
        "\"output_png\": \"viz/inline_cif.png\"}.\n\n"
        "Motif usage (JSON, interoperable with MotifDecomposition / motif libraries):\n"
        "  {\"mode\": \"motif\", \"motif\": { ...MotifPattern... }, "
        "\"output_png\": \"viz/my_motif.png\"}\n"
        "or\n"
        "  {\"mode\": \"motif\", \"motif_library_path\": \"path/to/motifs.json\", "
        "\"motif_name\": \"BC1O2_C0\", \"geometry\": \"auto\", "
        "\"output_png\": \"viz/BC1O2_C0.png\"}.\n"
        "This allows the agent to render multiple COFs and motifs in a single run "
        "without overwriting PNGs."
    )

    # Optional configuration
    vesta_exe: Optional[str] = None
    output_png: str = "viz/vastra_output.png"
    scale: int = 2
    nogui: bool = True

    def __init__(
        self,
        vesta_exe: Optional[str] = None,
        output_png: str = "viz/vastra_output.png",
        scale: int = 2,
        nogui: bool = True,
    ):
        super().__init__()
        self.vesta_exe = vesta_exe
        self.output_png = output_png
        self.scale = scale
        self.nogui = nogui

    # --------------- internal mode dispatch --------------- #

    def _run_motif_mode(self, payload: dict) -> str:
        """
        Handle JSON input with 'mode': 'motif'.

        Supports either an inline 'motif' dict or a pair
        (motif_library_path, motif_name). Allows per-call 'output_png'.
        """
        geometry = payload.get("geometry", "auto")
        out_png = payload.get("output_png", self.output_png)
        out_png_path = Path(out_png)

        motif: Optional[dict] = None

        if "motif" in payload and isinstance(payload["motif"], dict):
            motif = payload["motif"]
        else:
            lib_path = payload.get("motif_library_path", None)
            motif_name = payload.get("motif_name", None)
            if not lib_path or not motif_name:
                return (
                    "Vastra (motif mode): missing 'motif' or "
                    "'motif_library_path' + 'motif_name' in JSON."
                )
            try:
                motif = _load_motif_from_library(Path(lib_path), motif_name)
            except Exception as e:
                return f"Vastra (motif mode): failed to load motif from library: {e}"

        # Basic validation (MotifPattern schema)
        required_keys = {
            "name",
            "central_species",
            "neighbor_species_counts",
            "max_distance",
        }
        missing = [k for k in required_keys if k not in motif]
        if missing:
            return (
                "Vastra (motif mode): motif is missing required keys: "
                + ", ".join(missing)
            )

        try:
            cif_text = _build_cif_for_motif(motif, geometry=geometry)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                cif_path = _write_temp_cif_from_text(cif_text, tmpdir)
                png_path = _run_vesta_export(
                    cif_path=cif_path,
                    out_png=out_png_path,
                    vesta_exe=self.vesta_exe,
                    scale=self.scale,
                    nogui=self.nogui,
                )

            return f"Saved structure image to: {png_path}"
        except Exception as e:
            return f"Vastra/VESTA motif visualisation failed: {e}"

    def _run_cif_json_mode(self, payload: dict) -> str:
        """
        Handle JSON input for CIF visualisation with optional 'output_png'.
        Expected keys:
          - mode: 'cif' (optional but recommended)
          - cif_path: path to .cif   OR
          - cif_text: raw CIF text
          - output_png: PNG path override (optional)
        """
        out_png = payload.get("output_png", self.output_png)
        out_png_path = Path(out_png)

        cif_text: Optional[str] = None
        cif_path_str: Optional[str] = None

        if "cif_path" in payload:
            cif_path_str = payload["cif_path"]
        elif "path" in payload:
            cif_path_str = payload["path"]

        if cif_path_str:
            potential_path = Path(cif_path_str)
            if not potential_path.exists():
                return f"Vastra (cif mode): CIF path does not exist: {potential_path}"
            cif_path = potential_path.resolve()
            try:
                png_path = _run_vesta_export(
                    cif_path=cif_path,
                    out_png=out_png_path,
                    vesta_exe=self.vesta_exe,
                    scale=self.scale,
                    nogui=self.nogui,
                )
                return f"Saved structure image to: {png_path}"
            except Exception as e:
                return f"Vastra/VESTA visualisation failed (JSON cif mode, path): {e}"

        if "cif_text" in payload:
            cif_text = payload["cif_text"]

        if cif_text is None:
            return (
                "Vastra (cif mode): JSON payload must contain either 'cif_path' or "
                "'cif_text'."
            )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                cif_path = _write_temp_cif_from_text(cif_text, tmpdir)
                png_path = _run_vesta_export(
                    cif_path=cif_path,
                    out_png=out_png_path,
                    vesta_exe=self.vesta_exe,
                    scale=self.scale,
                    nogui=self.nogui,
                )
            return f"Saved structure image to: {png_path}"
        except Exception as e:
            return f"Vastra/VESTA visualisation failed (JSON cif mode, text): {e}"

    # --------------- main entrypoint --------------- #

    def _run(self, cif_input: str) -> str:
        """
        cif_input: either
          - CIF path or raw CIF text (legacy string mode), OR
          - JSON string with 'mode': 'motif' for motif visualisation, OR
          - JSON string describing CIF mode with optional 'output_png'.
        """
        cif_input = cif_input.strip()

        # Try JSON first – this unlocks motif mode and CIF-json mode
        payload = None
        if cif_input.startswith("{") or cif_input.startswith("["):
            try:
                payload = json.loads(cif_input)
            except json.JSONDecodeError:
                payload = None

        if isinstance(payload, dict):
            mode = payload.get("mode")
            if mode == "motif":
                return self._run_motif_mode(payload)
            # Treat anything that looks like CIF JSON as CIF mode
            if mode == "cif" or "cif_path" in payload or "cif_text" in payload:
                return self._run_cif_json_mode(payload)

        # Fall back to original CIF behaviour (string path or raw CIF text)
        potential_path = Path(cif_input)

        try:
            if potential_path.suffix.lower() == ".cif" and potential_path.exists():
                # Interpret as file path
                try:
                    cif_text = potential_path.read_text()
                except Exception as e:
                    return f"Vastra: failed to read CIF file '{potential_path}': {e}"
            else:
                # Interpret as raw CIF text
                cif_text = cif_input

            out_png_path = Path(self.output_png)

            if potential_path.suffix.lower() == ".cif" and potential_path.exists():
                # Use the original path directly
                cif_path = potential_path.resolve()
                png_path = _run_vesta_export(
                    cif_path=cif_path,
                    out_png=out_png_path,
                    vesta_exe=self.vesta_exe,
                    scale=self.scale,
                    nogui=self.nogui,
                )
            else:
                # Raw text → temp file
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    cif_path = _write_temp_cif_from_text(cif_text, tmpdir)
                    png_path = _run_vesta_export(
                        cif_path=cif_path,
                        out_png=out_png_path,
                        vesta_exe=self.vesta_exe,
                        scale=self.scale,
                        nogui=self.nogui,
                    )

            return f"Saved structure image to: {png_path}"
        except Exception as e:
            return f"Vastra/VESTA visualisation failed: {e}"

    async def _arun(self, cif_input: str) -> str:
        """Async use is not implemented for this tool."""
        raise NotImplementedError("This tool does not support async.")

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain.tools import BaseTool


# -------------------- VESTA helpers (based on qlip.visualization.vesta) -------------------- #

def _resolve_vesta(exe_hint: Optional[str] = None) -> str:
    """
    Resolve the VESTA executable path.

    Priority:
      1. explicit exe_hint argument (if given and exists)
      2. VESTA_EXE environment variable (if set and exists)
      3. shutil.which("VESTA") / shutil.which("VESTA.exe")
      4. (optional) a common default installation path on Windows

    This is modelled on qlip.visualization.vesta._resolve_vesta, with a couple of
    extra robustness tweaks for ChemCrow usage.
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
    Mirrors qlip.visualization.vesta._norm.
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

    NOTE: We *do not* use '-nogui' here because it crashes on this VESTA build.
    We just mimic the working CLI call:

        VESTA.exe -open <cif> -export_img scale=2 <png>
    """
    exe = _resolve_vesta(vesta_exe)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Build the exact command that works on your machine
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

    if result.returncode != 0:
        raise RuntimeError(
            "VESTA command failed with code {code}.\n"
            "CMD:\n{cmd}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(cmd),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )

    if not out_png.exists():
        raise RuntimeError(
            f"VESTA reported success but '{out_png}' was not created."
        )

    return str(out_png.resolve())



# ----------------------- ChemCrow / LangChain tool ----------------------- #

class VastraVisualise(BaseTool):

    """
    ChemCrow tool: call VESTA from the command line to render a CIF as a PNG.

    Input contract:
      - Either a path to an existing '.cif' file, OR
      - The raw CIF text content.

    Behaviour:
      - If the input string looks like a valid CIF path, it uses that file directly.
      - Otherwise, it writes the provided text to a temporary CIF file.
      - It then calls VESTA (resolved via 'vesta_exe', $VESTA_EXE or PATH) with
        '-open <cif> -export_img scale=<scale> <png>' to generate a PNG image.

    Why this is useful:
      - It gives the agent a quick way to produce a human-viewable picture of
        a proposed or retrieved structure without hand-opening VESTA.
      - You can visually sanity-check motif decompositions, connectivity, pore
        shapes, or suspected errors in the CIF.
      - Downstream systems (e.g. a notebook or UI) can display the PNG path
        returned by this tool.

    Requirements / limitations:
      - VESTA must be installed on the host machine and discoverable
        (via 'vesta_exe', $VESTA_EXE, or PATH).
      - This produces a static PNG; it does not expose any interactive VESTA
        features or 3D controls.
    """

    name = "VastraVisualise"
    description = (
        "Render a crystal structure from CIF into a PNG using VESTA. "
        "Call this tool with either a CIF filepath or the full CIF text. "
        "The tool resolves the VESTA executable, exports an image via the "
        "VESTA CLI, and saves it to 'viz/vastra_output.png' (or a configured path). "
        "Use it when you want a quick visual check of a structure (e.g. motif "
        "arrangement, pore geometry, obvious CIF mistakes) without manually "
        "opening VESTA. The output is a string giving the absolute PNG path."
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

    def _run(self, cif_input: str) -> str:
        """
        cif_input: either CIF text or a path to a .cif file.

        Heuristic, identical idea to what we discussed:
          - If cif_input is a path to an existing .cif file, read that file.
          - Otherwise, treat cif_input as raw CIF contents.
        """
        cif_input = cif_input.strip()
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

            # Use a temp dir for the intermediate cif if we are using raw text
            if potential_path.suffix.lower() == ".cif" and potential_path.exists():
                # Use the original path directly
                cif_path = potential_path
                # But still normalise/ensure absolute
                cif_path = cif_path.resolve()
                # run VESTA
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

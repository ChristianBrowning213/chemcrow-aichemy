# clean_chemcrow_minimal.py

import os
from pathlib import Path
from typing import List as TList

from chemcrow.agents import make_tools, ChemCrow
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

# === IMPORT YOUR CUSTOM TOOLS FROM tools/New ===
from chemcrow.tools.New.Arxiv2ResultLLM import Arxiv2ResultLLM
from chemcrow.tools.New.motif_tools import MotifDecompositionTool, MotifComparisonTool
from chemcrow.tools.New.VastraVisualise import VastraVisualise


# === HARD-CODED PATHS FOR YOUR SETUP ===

DEFAULT_VESTA = Path(r"C:\Users\brown\Documents\VESTA-win64\VESTA.exe")

DEFAULT_CRYSTAL_DIR = Path(
    r"C:\Users\brown\Documents\PhD\Winter School\COF_crystals\crystals"
)

TARGET_CIF = Path(
    r"C:\Users\brown\Documents\PhD\Winter School\COF_crystals\crystals\07000N2_ddec.cif"
)

# Your simple COF motif library
DEFAULT_MOTIF_LIB = Path(r"chemcrow\tools\New\motifs_cof_from_library.json")

os.environ.setdefault("VESTA_EXE", str(DEFAULT_VESTA))



class CheckCrystalFileTool(BaseTool):
    """
    Small tool that **does not** dump the whole directory,
    it only checks if a given CIF filename exists in DEFAULT_CRYSTAL_DIR.

    Input:
        query: either a bare filename like "07000N2_ddec.cif"
               or a full/absolute path (we take the .name).

    Output:
        A short string indicating whether the file exists and its full path.
    """

    name = "CheckCrystalFile"
    description = (
        f"Check whether a given CIF file exists in the default crystals directory "
        f"({DEFAULT_CRYSTAL_DIR}). Input can be just the filename "
        f"(e.g. '07000N2_ddec.cif') or a full path; the tool extracts the name."
    )

    crystals_dir: Path = DEFAULT_CRYSTAL_DIR

    def __init__(self, crystals_dir: Path | None = None):
        super().__init__()
        self.crystals_dir = crystals_dir or DEFAULT_CRYSTAL_DIR

    def _run(self, query: str) -> str:
        if not self.crystals_dir.exists():
            return f"Crystal directory not found: {self.crystals_dir}"

        if not query:
            return "No filename given. Please provide something like '07000N2_ddec.cif'."

        # If user passes a full path, strip to just the filename
        candidate_name = Path(query).name
        candidate_path = self.crystals_dir / candidate_name

        if candidate_path.exists():
            return f"FOUND: {candidate_name} at '{candidate_path}'."
        else:
            return f"NOT FOUND: {candidate_name} in '{self.crystals_dir}'."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async.")

def build_clean_chemcrow():
    tools_llm = ChatOpenAI(
        model_name=os.environ.get("CHEMCROW_TOOLS_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )

    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "SEMANTIC_SCHOLAR_API_KEY": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
    }

    all_tools = make_tools(tools_llm, api_keys=api_keys)

    allowed_stock = {
        "Python_REPL",
        "Wikipedia",
        "Mol2CAS",
        "PatentCheck",
        "SMILES2Weight",
        "FunctionalGroups",
    }

    clean_tools: TList[BaseTool] = [t for t in all_tools if t.name in allowed_stock]

    openai_key = api_keys["OPENAI_API_KEY"]

    # 1) Arxiv literature search
    arxiv_tool = Arxiv2ResultLLM(
        llm=tools_llm,
        openai_api_key=openai_key,
        max_results=20,
    )

    # 2) Motif decomposition & comparison
    motif_tool = MotifDecompositionTool(
        default_motif_library_path=str(DEFAULT_MOTIF_LIB)
    )

    motif_compare_tool = MotifComparisonTool(
        default_motif_library_path=str(DEFAULT_MOTIF_LIB)
    )

    # 3) VESTA visualisation (CIF → PNG)
    vastra_tool = VastraVisualise(
        vesta_exe=str(DEFAULT_VESTA),
        output_png="viz/vastra_output.png",
        scale=2,
        nogui=True,
    )

    # 4) CIF directory listing tool
    check_cif_tool = CheckCrystalFileTool(crystals_dir=DEFAULT_CRYSTAL_DIR)

    custom_tools: TList[BaseTool] = [
        arxiv_tool,
        motif_tool,
        motif_compare_tool,
        vastra_tool,
        check_cif_tool,
    ]

    clean_tools.extend(custom_tools)

    print("\nLoaded tools in CLEAN ChemCrow:")
    for t in clean_tools:
        print(f"  - {t.name}")

    chem_model = ChemCrow(
        tools=clean_tools,
    )

    return chem_model

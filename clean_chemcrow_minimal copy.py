# clean_chemcrow_minimal.py
"""
Clean ChemCrow instance that only loads the tools we consider “safe” plus our custom tools:
  - SAFE STOCK TOOLS:
      * Python_REPL
      * Wikipedia
      * Mol2CAS
      * PatentCheck
      * SMILES2Weight
      * FunctionalGroups

  - CUSTOM TOOLS:
      * Arxiv2ResultLLM (ArxivLiteratureSearch)
      * MotifDecompositionTool (MotifDecomposition)
      * VastraVisualise (VESTA CIF → PNG)
      * ListCrystalCIFsTool (ls/dir over CIF directory)

Everything with broken behaviour or MorganGenerator deprecation warnings
(Name2SMILES, SMILES2Name, MolSimilarity, ControlChemCheck, SimilarityToControlChem,
ExplosiveCheck, SafetySummary) is removed.
"""

import os
from pathlib import Path
from typing import List as TList

from chemcrow.agents import make_tools, ChemCrow
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

# === IMPORT YOUR CUSTOM TOOLS FROM tools/New ===
from chemcrow.tools.New.Arxiv2ResultLLM import Arxiv2ResultLLM
from chemcrow.tools.New.motif_tools import MotifDecompositionTool
from chemcrow.tools.New.VastraVisualise import VastraVisualise


# === HARD-CODED PATHS FOR YOUR SETUP ===

# VESTA installation
DEFAULT_VESTA = Path(r"C:\Users\brown\Documents\VESTA-win64\VESTA.exe")

# Directory with your COF CIFs
DEFAULT_CRYSTAL_DIR = Path(
    r"C:\Users\brown\Documents\PhD\Winter School\COF_crystals\crystals"
)

# Let VESTA_EXE be visible to the Vastra code if it looks for it
os.environ.setdefault("VESTA_EXE", str(DEFAULT_VESTA))


# === LOCAL TOOL: LIST CIF FILES IN YOUR CRYSTAL DIRECTORY ===

class ListCrystalCIFsTool(BaseTool):
    """
    Simple tool that lists all .cif files in DEFAULT_CRYSTAL_DIR.

    Input:
        query: ignored (can be an empty string).

    Output:
        A newline-separated list of CIF filenames, or an error message.
    """

    name = "ListCrystalCIFs"
    description = (
        f"List available CIF files in the default crystals directory "
        f"({DEFAULT_CRYSTAL_DIR}). "
        "Call this first if you want to know which CIFs you can visualise or analyse."
    )

    crystals_dir: Path = DEFAULT_CRYSTAL_DIR

    def __init__(self, crystals_dir: Path | None = None):
        super().__init__()
        self.crystals_dir = crystals_dir or DEFAULT_CRYSTAL_DIR

    def _run(self, query: str = "") -> str:
        if not self.crystals_dir.exists():
            return f"Crystal directory not found: {self.crystals_dir}"

        cifs: TList[str] = sorted(p.name for p in self.crystals_dir.glob("*.cif"))
        if not cifs:
            return f"No CIF files found in {self.crystals_dir}"

        return "\n".join(cifs)

    async def _arun(self, query: str = "") -> str:
        raise NotImplementedError("This tool does not support async.")


def build_clean_chemcrow():
    # LLM used *inside* the tools (same as ChemCrow uses internally)
    tools_llm = ChatOpenAI(
        model_name=os.environ.get("CHEMCROW_TOOLS_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )

    # Keys ChemCrow expects
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "SEMANTIC_SCHOLAR_API_KEY": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
    }

    # Build all stock tools
    all_tools = make_tools(tools_llm, api_keys=api_keys)

    # Whitelist only the "safe" stock tools
    allowed_stock = {
        "Python_REPL",
        "Wikipedia",
        "Mol2CAS",
        "PatentCheck",
        "SMILES2Weight",
        "FunctionalGroups",
    }

    clean_tools: TList[BaseTool] = [t for t in all_tools if t.name in allowed_stock]

    # ---- ADD YOUR CUSTOM TOOLS EXPLICITLY ----

    openai_key = api_keys["OPENAI_API_KEY"]

    # 1) Arxiv literature search
    arxiv_tool = Arxiv2ResultLLM(
        llm=tools_llm,
        openai_api_key=openai_key,
        max_results=20,
    )

    # 2) Motif decomposition (CIF motif analysis)
    motif_tool = MotifDecompositionTool()

    # 3) VESTA visualisation (CIF → PNG)
    vastra_tool = VastraVisualise(
        vesta_exe=str(DEFAULT_VESTA),
        output_png="viz/vastra_output.png",
        scale=2,
        nogui=True,
    )

    # 4) CIF directory listing tool
    list_cifs_tool = ListCrystalCIFsTool(crystals_dir=DEFAULT_CRYSTAL_DIR)

    custom_tools: TList[BaseTool] = [
        arxiv_tool,
        motif_tool,
        vastra_tool,
        list_cifs_tool,
    ]

    # Combine stock + custom
    clean_tools.extend(custom_tools)

    print("\nLoaded tools in CLEAN ChemCrow:")
    for t in clean_tools:
        print(f"  - {t.name}")

    # Build ChemCrow using ONLY these tools
    chem_model = ChemCrow(
        tools=clean_tools,  # <— this is the important bit
        # leave everything else at default so we don't guess wrong
    )

    return chem_model


if __name__ == "__main__":
    print("Building clean ChemCrow instance (no deprecated / broken tools, plus custom tools)...")
    chem_model = build_clean_chemcrow()

    # Also list tools in a more detailed way
    print("\n=== ChemCrow tools (CLEAN + CUSTOM) ===\n")
    tools_obj = chem_model.tools if hasattr(chem_model, "tools") else []

    for idx, tool in enumerate(tools_obj, start=1):
        name = getattr(tool, "name", repr(tool))
        desc = getattr(tool, "description", "(no description)")
        print(f"{idx}. {name}")
        print(f"   {desc}")
        print()

    # Optional quick test
    # question = "List the CIF files you can see, then pick one and visualise it."
    # print("\n=== TEST QUERY ===")
    # print(f"Q: {question}\n")
    # answer = chem_model.run(question)
    # print("A:", answer)

# test_all_tools.py
"""
End-to-end test script that tells ChemCrow to exercise all custom tools:

  - CheckCrystalFile
  - VastraVisualise
  - MotifDecomposition
  - MotifComparison
  - ArxivLiteratureSearch
  - (optionally) COFMultiObjectiveBO

Assumes:
  - clean_chemcrow_minimal.py defines:
        build_clean_chemcrow
        DEFAULT_CRYSTAL_DIR
        TARGET_CIF
        DEFAULT_MOTIF_LIB
  - motifs_cof_from_library.json exists at DEFAULT_MOTIF_LIB.
  - If you want the BO step, COFMultiObjectiveBO is imported and registered
    as a tool inside build_clean_chemcrow.

COF BO dataset is assumed to live at:

  CIF directory:
    C:\\Users\\brown\\Downloads\\COF_crystals\\crystals

  Descriptor CSV:
    C:\\Users\\brown\\Downloads\\COF_crystals\\cof_descriptors.csv

  Property CSV:
    C:\\Users\\brown\\Downloads\\COF_crystals\\gcmc_calculations.csv
"""

import os
import json

from clean_chemcrow_minimal import (
    build_clean_chemcrow,
    DEFAULT_CRYSTAL_DIR,
    TARGET_CIF,
    DEFAULT_MOTIF_LIB,
)


# --- COF BO dataset paths (Windows) --- #

COF_CIF_DIR = r"C:\Users\brown\Downloads\COF_crystals\crystals"
COF_DESCRIPTOR_CSV = r"C:\Users\brown\Downloads\COF_crystals\cof_descriptors.csv"
COF_PROPERTY_CSV = r"C:\Users\brown\Downloads\COF_crystals\gcmc_calculations.csv"

# Just the filename of the target CIF, e.g. "07000N2_ddec.cif"
target_cif_name = TARGET_CIF.name


def main():
    # Make sure OpenAI key is visible
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()

    # Which tools are actually registered?
    tool_names = {t.name for t in getattr(chem_model, "tools", [])}
    has_cof_mobo = "COFMultiObjectiveBO" in tool_names

    print("\n=== TOOLS SEEN BY ChemCrow ===")
    for n in sorted(tool_names):
        print(f"  - {n}")
    print(f"\nCOFMultiObjectiveBO registered? {has_cof_mobo}\n")

    # Strings for prompt context (for human, not for JSON)
    target_cif_str = str(TARGET_CIF)
    crystal_dir_str = str(DEFAULT_CRYSTAL_DIR)
    motif_lib_str = str(DEFAULT_MOTIF_LIB)

    # JSON payload for MotifDecomposition.
    # We only pass the CIF NAME; the tool resolves it using its default CIF dir
    # and uses its default motif library path.
    motif_decomp_payload = {
        "mode": "all",
        "cif_name": target_cif_name,
        "allow_overlap": True,
    }
    motif_decomp_json = json.dumps(motif_decomp_payload)

    # --- JSON payload for COFMultiObjectiveBO (multi-objective BO) --- #

    cof_mobo_config_json = ""
    bo_section = ""
    extra_tool_line = ""

    if has_cof_mobo:
        # For the BO tool we still pass full paths and escape backslashes,
        # because that tool expects raw paths in its JSON.
        cof_cif_dir_json = COF_CIF_DIR.replace("\\", "\\\\")
        cof_desc_csv_json = COF_DESCRIPTOR_CSV.replace("\\", "\\\\")
        cof_prop_csv_json = COF_PROPERTY_CSV.replace("\\", "\\\\")

        # We maximise two objectives:
        #   1. "⟨N⟩ (mmol/g)"      – uptake per gram
        #   2. "selectivity Xe/Kr" – separation performance
        cof_mobo_config_json = (
            "{"
            f"\"cif_dir\": \"{cof_cif_dir_json}\", "
            f"\"descriptor_csv\": \"{cof_desc_csv_json}\", "
            f"\"property_csvs\": [\"{cof_prop_csv_json}\"], "
            "\"target_properties\": ["
            "\"⟨N⟩ (mmol/g)\", "
            "\"selectivity Xe/Kr\""
            "], "
            "\"id_column\": \"crystal_name\", "
            "\"property_agg\": \"mean\", "
            "\"top_k\": 15, "
            "\"n_weight_samples\": 6"
            "}"
        )

        extra_tool_line = "  - COFMultiObjectiveBO\n"

        bo_section = f"""
6) Call **COFMultiObjectiveBO** once on the COF dataset, using EXACTLY the
   following JSON string as the tool input (do NOT modify this string):

   {cof_mobo_config_json}

   This JSON configures multi-objective BO to MAXIMISE two objectives:

     - "⟨N⟩ (mmol/g)"      (uptake per gram)
     - "selectivity Xe/Kr" (Xe/Kr separation performance)

   After the tool returns, summarise:
   - How many COFs were considered.
   - Which COFs are currently on the observed Pareto front (list the crystal id
     and values for both objectives).
   - The top BO suggestions (up to 15), with their predicted values, uncertainties,
     and EI-based ranking.
"""

    # Build the natural-language prompt for the agent
    test_prompt = f"""
You are wired into a ChemCrow environment with the following tools available
(at least):

  - Python_REPL
  - Wikipedia
  - Mol2CAS
  - PatentCheck
  - SMILES2Weight
  - FunctionalGroups

  - CheckCrystalFile
  - VastraVisualise
  - MotifDecomposition
  - MotifComparison
  - ArxivLiteratureSearch
{extra_tool_line.rstrip()}

Your job is to exercise ALL of the custom tools at least once in a sensible way.

The CIF directory for the motif tools is:

  {crystal_dir_str}

A specific CIF of interest is:

  {target_cif_str}

The motif library (simple COF motifs) is at:

  {motif_lib_str}

Additionally, there is a separate COF dataset for Bayesian optimisation:

  COF CIF directory:
    {COF_CIF_DIR}

  COF descriptor CSV:
    {COF_DESCRIPTOR_CSV}

  COF property CSV:
    {COF_PROPERTY_CSV}

Follow these steps, using tool calls explicitly:

1) Use **CheckCrystalFile** to verify that the CIF
   "07000N2_ddec.cif"
   exists in the default crystal directory. For later steps that need a full
   path (e.g. VESTA visualisation), use the absolute path returned by this tool.

2) Call **MotifDecomposition** once on that CIF with EXACTLY the following JSON
   string as the input:

   {motif_decomp_json}

   This JSON uses the CIF filename ('cif_name'). The tool will resolve it inside
   the default crystal directory and will use its default motif library path.

   Then:
   - Summarise how many motif instances were found, grouped by motif_name.
   - State how many unassigned_sites there are.

3) Call **VastraVisualise** once on the SAME CIF, with the full absolute path
   as the input (e.g. "{target_cif_str}" or the path reported by CheckCrystalFile).
   Wait for the tool result. Tell me where the PNG was written (or report the
   error if VESTA fails).

4) If there is at least one OTHER CIF in the directory, call **MotifComparison**
   once, comparing the primary CIF from step (2) with a second CIF.
   Use a JSON input of the form:

   {{
     "cif_name_1": "07000N2_ddec.cif",
     "cif_name_2": "07010N3_ddec.cif",
     "allow_overlap": true
   }}

   The tool will resolve these filenames inside the default crystal directory
   and will use its default motif library path configured in the environment.

   Then:
   - List which motifs are shared between the two structures, with their counts.
   - List which motifs are unique to each structure.
   If you cannot easily identify a second CIF, explain this and skip this step.

5) Call **ArxivLiteratureSearch** to answer the following question:

   "cool mammal facts"

   Answer based ONLY on ArxivLiteratureSearch outputs. Summarise in 3–5 sentences.
"""

    print("\n=== TEST PROMPT (exercise all custom tools) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

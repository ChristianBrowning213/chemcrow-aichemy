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
  - motifs_cof_simple.json exists at DEFAULT_MOTIF_LIB.
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

    target_cif_str = str(TARGET_CIF)
    crystal_dir_str = str(DEFAULT_CRYSTAL_DIR)
    motif_lib_str = str(DEFAULT_MOTIF_LIB)

    # Windows JSON escaping for motif/CIF paths
    target_cif_json = target_cif_str.replace("\\", "\\\\")
    motif_lib_json = motif_lib_str.replace("\\", "\\\\")

    # JSON payload for MotifDecomposition
    motif_decomp_json = (
        "{"
        f"\"mode\": \"all\", "
        f"\"cif_path\": \"{target_cif_json}\", "
        f"\"motif_library_path\": \"{motif_lib_json}\", "
        "\"allow_overlap\": true"
        "}"
    )

    # --- JSON payload for COFMultiObjectiveBO (multi-objective BO) --- #

    cof_mobo_config_json = ""
    bo_section = ""
    extra_tool_line = ""

    if has_cof_mobo:
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
7) Call **COFMultiObjectiveBO** once on the COF dataset, using EXACTLY the
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
   exists in the default crystal directory. Use the full absolute path returned
   by this tool in all later steps.

2) Call **MotifDecomposition** once on that CIF with EXACTLY the following JSON
   string as the input (do not alter it other than inserting the correct slashes):

   {motif_decomp_json}

   Then:
   - Summarise how many motif instances were found, grouped by motif_name.
   - State how many unassigned_sites there are.

3) Call **VastraVisualise** once on the SAME CIF, with the full absolute path
   as the input (e.g. "{target_cif_str}").
   Wait for the tool result. Tell me where the PNG was written (or report the
   error if VESTA fails).

4) If there is at least one OTHER CIF in the directory, call **MotifComparison**
   once, comparing the primary CIF from step (2) with a second CIF.
   Use a JSON input of the form:

   {{
     "cif_path_1": "FULL_PATH_TO_FIRST_CIF",
     "cif_path_2": "FULL_PATH_TO_SECOND_CIF",
     "motif_library_path": "{motif_lib_str}",
     "allow_overlap": true
   }}

   Then:
   - List which motifs are shared between the two structures, with their counts.
   - List which motifs are unique to each structure.
   If you cannot easily identify a second CIF, explain this and skip this step.

5) Call **ArxivLiteratureSearch** once to answer the following question:

   "How are covalent organic frameworks (COFs) used for gas storage applications?"

   Answer based ONLY on ArxivLiteratureSearch outputs. Summarise in 3–5 sentences.
{bo_section}
8) At the end, briefly confirm which tools you actually invoked
   (CheckCrystalFile, MotifDecomposition, MotifComparison, VastraVisualise,
   ArxivLiteratureSearch{", COFMultiObjectiveBO" if has_cof_mobo else ""})
   and what each one returned at a high level.

Important:
- Do NOT fabricate tool outputs; rely entirely on actual tool calls.
- If any tool fails (e.g. missing file, VESTA not installed, motif library missing,
  COF CSVs missing, or wrong column names), clearly report the error message and
  continue with the remaining steps where possible.
"""

    print("\n=== TEST PROMPT (exercise all custom tools) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

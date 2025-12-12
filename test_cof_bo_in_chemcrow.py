# test_cof_bo_premerged_in_chemcrow.py
"""
Smoke test for COFMultiObjectiveBO using a pre-merged BO table
(crystal_selectivity_clean.csv) *inside* the ChemCrow agent.

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow and
    registers COFMultiObjectiveBO as a tool.
  - The cleaned table lives at:
        C:\\Users\\brown\\Downloads\\COF_crystals\\crystal_selectivity_clean.csv

This table has columns:
  - crystal_name
  - a bunch of descriptor columns (pore_diameter_Å, void_fraction, ...)
  - selectivity_Xe/Kr          (single-objective of interest)

For testing multi-objective BO, we use:
  - target 1: selectivity_Xe/Kr
  - target 2: surface_area_m²g⁻¹
"""

import os

from clean_chemcrow_minimal import build_clean_chemcrow


# --- Pre-merged BO table (Windows path) ------------------------------------ #

CLEAN_BO_CSV = r"C:\Users\brown\Downloads\COF_crystals\crystal_selectivity_clean.csv"


def main() -> None:
    # Warn if no key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()
    print("\n[DEBUG] ChemCrow agent constructed.\n")

    # Escape backslashes for JSON
    clean_csv_json = CLEAN_BO_CSV.replace("\\", "\\\\")

    # JSON payload for COFMultiObjectiveBO
    # We treat the same CSV as both descriptors and properties.
    cof_mobo_config_json = (
        "{"
        f"\"descriptor_csv\": \"{clean_csv_json}\", "
        f"\"property_csvs\": [\"{clean_csv_json}\"], "
        "\"cif_dir\": \"\", "
        "\"id_column\": \"crystal_name\", "
        "\"target_properties\": ["
        "\"selectivity_Xe/Kr\", "
        "\"surface_area_m²g⁻¹\""
        "], "
        "\"property_agg\": \"mean\", "
        "\"top_k\": 15, "
        "\"n_weight_samples\": 6"
        "}"
    )

    # Minimal natural-language prompt for the agent
    test_prompt = f"""
You are wired into a ChemCrow environment with the following tools available:

  - COFMultiObjectiveBO

Your task:

1) Call **COFMultiObjectiveBO** exactly once using EXACTLY the JSON string
   below as the tool input (do NOT modify this string):

   {cof_mobo_config_json}

2) When the tool returns, output ONLY the tool's text result.
   Do not add any extra commentary, explanation, or markdown.
"""

    print("\n=== TEST PROMPT (COFMultiObjectiveBO on pre-merged table) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

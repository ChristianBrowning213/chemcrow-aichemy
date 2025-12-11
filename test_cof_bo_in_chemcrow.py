# test_cof_bo_props_only_in_chemcrow.py
"""
Smoke test for InspectCSVDataset + COFMultiObjectiveBO *inside* the ChemCrow agent,
using ONLY the property CSV as both "descriptor" and "property" source.

For now we IGNORE the true descriptor CSV and just use:

  C:\\Users\\brown\\Downloads\\COF_crystals\\gcmc_calculations.csv

as:
  - descriptor_csv
  - property_csvs[0]

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow.
  - build_clean_chemcrow() registers:
        - InspectCSVDataset
        - COFMultiObjectiveBO
"""

import os

from clean_chemcrow_minimal import build_clean_chemcrow


# --- COF dataset paths (Windows) ------------------------------------------- #

COF_CIF_DIR = r"C:\Users\brown\Downloads\COF_crystals\crystals"
COF_PROPERTY_CSV = r"C:\Users\brown\Downloads\COF_crystals\gcmc_calculations.csv"


def main() -> None:
    # Warn if no key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()

    print("\n[DEBUG] ChemCrow agent constructed.\n")

    # Windows JSON escaping for paths
    cof_cif_dir_json = COF_CIF_DIR.replace("\\", "\\\\")
    cof_prop_csv_json = COF_PROPERTY_CSV.replace("\\", "\\\\")

    # For this smoke test, we use the property CSV as BOTH descriptor and property.
    descriptor_csv_json = cof_prop_csv_json

    # JSON payload for InspectCSVDataset
    inspect_config_json = (
        "{"
        f"\"descriptor_csv\": \"{descriptor_csv_json}\", "
        f"\"property_csvs\": [\"{cof_prop_csv_json}\"], "
        f"\"cif_dir\": \"{cof_cif_dir_json}\""
        "}"
    )

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
  - InspectCSVDataset
  - COFMultiObjectiveBO

For THIS SMOKE TEST, we deliberately ignore the separate descriptor CSV.
Instead, we use the SAME file

  C:\\Users\\brown\\Downloads\\COF_crystals\\gcmc_calculations.csv

as BOTH:
  - descriptor_csv
  - property_csvs[0]

The COF dataset lives at:

  COF CIF directory:
    {COF_CIF_DIR}

  Property CSV (also used as descriptor CSV in this test):
    {COF_PROPERTY_CSV}

We want to MAXIMISE two objectives jointly:

  1) "⟨N⟩ (mmol/g)"      – uptake per gram
  2) "selectivity Xe/Kr" – Xe/Kr separation performance

Follow these steps, using tool calls explicitly:

1) First, call **InspectCSVDataset** exactly once with the following JSON
   string as input (do NOT modify this string):

   {inspect_config_json}

2) From the InspectCSVDataset output, answer briefly:

   - How many rows and columns the CSV has.
   - Which column looks like a good ID column (for this file, this is expected
     to be 'xtal').
   - Confirm that the target_properties "⟨N⟩ (mmol/g)" and "selectivity Xe/Kr"
     exist and are numeric.

3) Using that information, construct a JSON config for **COFMultiObjectiveBO**
   that:

   - Uses this SAME path for descriptor and property:

        "descriptor_csv": "{COF_PROPERTY_CSV}"
        "property_csvs": ["{COF_PROPERTY_CSV}"]

   - Uses:

        "cif_dir": "{COF_CIF_DIR}"

   - Uses:

        "id_column": "xtal"

   - Uses the two target_properties:

        "⟨N⟩ (mmol/g)"
        "selectivity Xe/Kr"

   - Sets:

        "property_agg": "mean"
        "top_k": 15
        "n_weight_samples": 6

   Show me the JSON config you will use.

4) Call **COFMultiObjectiveBO** exactly once on the dataset, using the JSON
   config from step (3) as the tool input.

5) When COFMultiObjectiveBO returns, summarise in your own words:

   - How many COFs (rows) were considered in total.
   - How many rows are fully observed (all target_properties present).
   - The Pareto front: for each non-dominated COF, list:
       • the ID (xtal),
       • its values for "⟨N⟩ (mmol/g)" and "selectivity Xe/Kr".
   - The top BO suggestions (up to 15), as ranked by EI, including for each:
       • the ID (xtal),
       • predicted mean ± uncertainty for both objectives.

6) If COFMultiObjectiveBO reports an error, quote the error message and explain
   what appears to be wrong with the configuration or the data.
"""

    print("\n=== TEST PROMPT (InspectCSVDataset + COFMultiObjectiveBO, props-only) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

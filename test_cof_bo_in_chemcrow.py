# test_cof_bo_inspect_clean_in_chemcrow.py
"""
Smoke test for InspectCSVDataset + COFMultiObjectiveBO *inside* the ChemCrow agent,
including basic data cleaning to reconcile different ID columns.

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow.
  - build_clean_chemcrow() registers:
        - InspectCSVDataset
        - COFMultiObjectiveBO
  - Dataset paths (Windows):

      CIF directory:
        C:\\Users\\brown\\Downloads\\COF_crystals\\crystals

      Descriptor CSV:
        C:\\Users\\brown\\Downloads\\COF_crystals\\cof_descriptors.csv

      Property CSV:
        C:\\Users\\brown\\Downloads\\COF_crystals\\gcmc_calculations.csv
"""

import os

from clean_chemcrow_minimal import build_clean_chemcrow


# --- COF BO dataset paths (Windows) ---------------------------------------- #

COF_CIF_DIR = r"C:\Users\brown\Downloads\COF_crystals\crystals"
COF_DESCRIPTOR_CSV = r"C:\Users\brown\Downloads\COF_crystals\cof_descriptors.csv"
COF_DESCRIPTOR_CSV_CLEAN = r"C:\Users\brown\Downloads\COF_crystals\cof_descriptors_with_xtal.csv"
COF_PROPERTY_CSV = r"C:\Users\brown\Downloads\COF_crystals\gcmc_calculations.csv"


def main() -> None:
    # Warn if no key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()

    print("\n[DEBUG] ChemCrow agent constructed.\n")

    # Windows JSON escaping for paths
    cof_cif_dir_json = COF_CIF_DIR.replace("\\", "\\\\")
    cof_desc_csv_json = COF_DESCRIPTOR_CSV.replace("\\", "\\\\")
    cof_desc_clean_json = COF_DESCRIPTOR_CSV_CLEAN.replace("\\", "\\\\")
    cof_prop_csv_json = COF_PROPERTY_CSV.replace("\\", "\\\\")

    # JSON payload for InspectCSVDataset
    inspect_config_json = (
        "{"
        f"\"descriptor_csv\": \"{cof_desc_csv_json}\", "
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

Your job is to:

  (A) verify that the CSV inspection works and understand how the tables relate,
  (B) perform basic data cleaning to reconcile different ID columns, and
  (C) run multi-objective Bayesian optimisation on the COF dataset.

The COF dataset for Bayesian optimisation lives at:

  COF CIF directory:
    {COF_CIF_DIR}

  COF descriptor CSV:
    {COF_DESCRIPTOR_CSV}

  COF property CSV:
    {COF_PROPERTY_CSV}

We want to MAXIMISE two objectives jointly (with **selectivity** as the primary
design goal, but still considering uptake):

  1) "⟨N⟩ (mmol/g)"      – uptake per gram
  2) "selectivity Xe/Kr" – Xe/Kr separation performance

Follow these steps, using tool calls explicitly:

1) First, call **InspectCSVDataset** exactly once with the following JSON
   string as input (do NOT modify this string):

   {inspect_config_json}

2) Carefully read the InspectCSVDataset output and, based on it, answer:

   - Which ID-like columns exist in the DESCRIPTOR CSV (e.g. 'crystal_name')?
   - Which ID-like columns exist in the PROPERTY CSV (e.g. 'xtal')?
   - Explain briefly why the dataset currently cannot be merged on a single
     shared column name.

3) Perform **data cleaning and ID reconciliation** using **Python_REPL**:

   - Use pandas to:
       * load the descriptor CSV from "{COF_DESCRIPTOR_CSV}",
       * check that 'crystal_name' is present and inspect its uniqueness,
       * if there is no 'xtal' column in the descriptor DataFrame, create one:

           descriptor_df["xtal"] = descriptor_df["crystal_name"]

       * optionally strip whitespace from 'crystal_name' and 'xtal' to avoid
         subtle mismatches,
       * save the cleaned descriptor DataFrame to:

           "{COF_DESCRIPTOR_CSV_CLEAN}"

         with index=False.
   - After writing, print:
       * descriptor_df.shape,
       * list(descriptor_df.columns),
       * and a few example values of descriptor_df["xtal"].
   - Confirm explicitly (in words) that the cleaned descriptor CSV now has an
     'xtal' column that should match the 'xtal' column in the property CSV.

4) Construct a JSON config for **COFMultiObjectiveBO** that:

   - Uses the CLEANED descriptor CSV:

       "descriptor_csv": "{COF_DESCRIPTOR_CSV_CLEAN}"

   - Uses the SAME property CSV and CIF directory as above:

       "property_csvs": ["{COF_PROPERTY_CSV}"]
       "cif_dir": "{COF_CIF_DIR}"

   - Uses the unified ID column:

       "id_column": "xtal"

   - Uses the two target_properties:

       "target_properties": ["⟨N⟩ (mmol/g)", "selectivity Xe/Kr"]

   - Sets:

       "property_agg": "mean"
       "top_k": 15
       "n_weight_samples": 6

   Show me the final JSON config you will use as the tool input.

5) Call **COFMultiObjectiveBO** exactly once on the COF dataset, using the
   JSON config from step (4) as the tool input.

6) When COFMultiObjectiveBO returns, summarise in your own words:

   - How many COFs were considered in total.
   - How many COFs are currently fully observed (all target_properties present).
   - Which COFs are on the observed Pareto front. For each, list:
       • the crystal identifier (xtal),
       • its values for "⟨N⟩ (mmol/g)" and "selectivity Xe/Kr",
       • and emphasise which ones show the best **selectivity**.
   - The top BO suggestions (up to 15), as ranked by EI, including for each:
       • the crystal identifier,
       • predicted mean ± uncertainty for both objectives,
       • and a brief note on how good its selectivity is compared to others.

7) If COFMultiObjectiveBO reports an error (e.g. missing columns, not enough
   fully observed points), quote the error message and explain what appears to
   be wrong with the configuration or data, and suggest how the cleaning could
   be extended to fix it.
"""
    print("\n=== TEST PROMPT (InspectCSVDataset + data cleaning + COFMultiObjectiveBO) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

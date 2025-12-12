# test_generic_bo_in_chemcrow.py
"""
Smoke test for GenericBayesOpt1D *inside* the ChemCrow agent.

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow.
  - build_clean_chemcrow() registers GenericBayesOpt1D as a tool.
  - CSV path (Windows):

      C:\\Users\\brown\\Downloads\\COF_crystals\\crystal_selectivity_clean.csv

The CSV is assumed to contain:
  - an ID-like column (e.g. 'crystal_name'),
  - a scalar objective related to selectivity
    (e.g. 'selectivity_Xe/Kr' or similar),
  - other numeric descriptor columns.

This script asks the agent to:
  1) Call GenericBayesOpt1D exactly once on the CSV, using a JSON config.
  2) Let the tool introspect the CSV and resolve the objective column from a
     fuzzy goal string like "selectivity".
  3) Return a human-readable BO summary and print it.
"""

import os
import json

from clean_chemcrow_minimal import build_clean_chemcrow


CSV_PATH = r"C:\Users\brown\Downloads\COF_crystals\crystal_selectivity_clean.csv"


def main() -> None:
    # Warn if no key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()
    print("\n[DEBUG] ChemCrow agent constructed.\n")

    # Build JSON config for GenericBayesOpt1D
    config = {
        "csv_path": CSV_PATH,
        # Fuzzy goal label; the tool will resolve to the actual column name
        # (e.g. 'selectivity_Xe/Kr', 'selectivity_XeKr', etc.)
        "goal": "selectivity",
        # ID column can be omitted and inferred, but we can also be explicit:
        "id_column": "crystal_name",
        "maximise": True,
        "top_k": 10,
    }
    config_json = json.dumps(config)

    # Windows-friendly JSON string for display in the prompt
    # (no need to escape backslashes manually because json.dumps already did).
    # We will just embed this as-is.
    test_prompt = f"""
You are wired into a ChemCrow environment with the following tools available
(at least):

  - Python_REPL
  - Wikipedia
  - Mol2CAS
  - PatentCheck
  - SMILES2Weight
  - FunctionalGroups
  - GenericBayesOpt1D

We have a single CSV that already contains both descriptors and a selectivity
objective:

  COF CSV (descriptors + selectivity):
    {CSV_PATH}

Your job is to run 1D Bayesian optimisation over this CSV using
**GenericBayesOpt1D** and then summarise the result.

Follow these steps, using tool calls explicitly:

1) Call **GenericBayesOpt1D** exactly once with the following JSON config
   string as input (do NOT modify this string):

   {config_json}

2) Wait for the tool result and then summarise, in your own words:

   - Which column the tool resolved as the objective (selectivity).
   - Which column it used as the ID.
   - How many rows in total the CSV had.
   - How many rows had observed (finite) objective values versus how many were
     treated as 'unmeasured' candidates.
   - The top 10 suggested IDs, including for each:
       • the ID,
       • the predicted mean ± uncertainty for the objective,
       • the EI score (if available).

3) If the tool reports an error (e.g. CSV not found, no numeric columns, could
   not resolve the goal to a column), quote the error message and explain what
   appears to be wrong with the configuration or the CSV structure.
"""

    print("\n=== TEST PROMPT (exercise GenericBayesOpt1D) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()

# test_all_tools.py
"""
End-to-end test script that tells ChemCrow to exercise all custom tools:

  - ListCrystalCIFs
  - VastraVisualise
  - MotifDecomposition
  - MotifComparison
  - ArxivLiteratureSearch

Assumes:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow, DEFAULT_CRYSTAL_DIR, TARGET_CIF.
  - motifs_cof_simple.json exists at DEFAULT_MOTIF_LIB.
"""

import os

from clean_chemcrow_minimal import (
    build_clean_chemcrow,
    DEFAULT_CRYSTAL_DIR,
    TARGET_CIF,
    DEFAULT_MOTIF_LIB,
)


def main():
    # Make sure OpenAI key is visible
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    chem_model = build_clean_chemcrow()

    target_cif_str = str(TARGET_CIF)
    crystal_dir_str = str(DEFAULT_CRYSTAL_DIR)
    motif_lib_str = str(DEFAULT_MOTIF_LIB)

    # Windows JSON escaping for paths
    target_cif_json = target_cif_str.replace("\\", "\\\\")
    motif_lib_json = motif_lib_str.replace("\\", "\\\\")

    # JSON payload examples we want the agent to send
    motif_decomp_json = (
        "{"
        f"\"mode\": \"all\", "
        f"\"cif_path\": \"{target_cif_json}\", "
        f"\"motif_library_path\": \"{motif_lib_json}\", "
        "\"allow_overlap\": true"
        "}"
    )

    motif_compare_json_template = (
        "{{"
        "\"cif_path_1\": \"{cif1}\", "
        "\"cif_path_2\": \"{cif2}\", "
        f"\"motif_library_path\": \"{motif_lib_json}\", "
        "\"allow_overlap\": true"
        "}}"
    )

    test_prompt = f"""

 Call **ArxivLiteratureSearch** to answer the following question:

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

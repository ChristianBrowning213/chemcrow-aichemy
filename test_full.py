# test_selectivity_motif_pipeline_robust.py
"""
Robust end-to-end test for the COF selectivity + motifs pipeline inside CLEAN ChemCrow.

Goal (in words):
  "We want to find whether COFs in this dataset have above-average or at least
   notable Xe/Kr selectivity, identify the top-performing motifs shared by the
   best crystals, visualise both the high-performing COFs and their key motifs,
   and produce a machine-readable summary of the results."

The agent should:
  - Use Arxiv2ResultLLM to establish a literature benchmark for 'high' Xe/Kr
    selectivity in MOFs/COFs.
  - Use InspectCSVDataset to understand the structure of the COF selectivity CSV.
  - Use GenericBayesOpt1D to perform 1D BO on selectivity and find the top-10
    high-selectivity COFs.
  - For those top-10 COFs, run motif decomposition and identify motifs that
    occur frequently among them.
  - Visualise a few of the best COFs that share many of the top motifs,
    using per-call output filenames.
  - Visualise 2–3 of the most frequent motifs themselves (as isolated
    coordination environments) using VastraVisualise in motif mode, again
    with per-call output filenames.
  - Summarise which motifs seem most associated with high selectivity, how
    'notable' those selectivities are compared to the literature benchmark,
    and store all key IDs / names / PNG paths in a JSON block.

Assumptions:
  - clean_chemcrow_minimal.py defines build_clean_chemcrow and constants:
      DEFAULT_CRYSTAL_DIR
      DEFAULT_MOTIF_LIB
  - The following tools are registered with those names:
      - "ArxivLiteratureSearch" (Arxiv2ResultLLM wrapper)
      - "InspectCSVDataset"
      - "GenericBayesOpt1D"
      - "MotifDecomposition"
      - "VastraVisualise"
      - "CheckCrystalFile"
      - "Python_REPL"
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from clean_chemcrow_minimal import (
    build_clean_chemcrow,
    DEFAULT_CRYSTAL_DIR,
    DEFAULT_MOTIF_LIB,
)

# Path to your pre-cleaned selectivity CSV
CSV_PATH = r"C:\Users\brown\Downloads\COF_crystals\crystal_selectivity_clean.csv"


def setup_logging() -> logging.Logger:
    """
    Configure logging so that everything goes both to:
      - stdout (for live viewing)
      - a timestamped log file under ./logs
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"selectivity_motif_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),  # stdout
        ],
    )

    logger = logging.getLogger("selectivity_motif_pipeline")
    logger.info("Logging initialised.")
    logger.info(f"Log file: {log_path}")
    return logger


def main() -> None:
    logger = setup_logging()

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set. Tools that call OpenAI may fail.\n")

    logger.info("Building CLEAN ChemCrow agent...")
    chem_model = build_clean_chemcrow()
    logger.info("ChemCrow agent constructed.")

    # JSON config for InspectCSVDataset
    inspect_config = {
        "descriptor_csv": CSV_PATH,
        "property_csvs": [CSV_PATH],
        "cif_dir": str(DEFAULT_CRYSTAL_DIR),
    }
    inspect_config_json = json.dumps(inspect_config)

    # JSON config for GenericBayesOpt1D
    generic_bo_config = {
        "csv_path": CSV_PATH,
        "goal": "selectivity",   # fuzzy label; tool must map to a real column
        "top_k": 10,
    }
    generic_bo_config_json = json.dumps(generic_bo_config)

    # Convert paths to strings for embedding in the prompt
    cif_dir_str = str(DEFAULT_CRYSTAL_DIR)
    motif_lib_str = str(DEFAULT_MOTIF_LIB)

    test_prompt = f"""
        You are wired into a CLEAN ChemCrow environment with at least the following tools:

        - Python_REPL
        - Wikipedia
        - Mol2CAS
        - PatentCheck
        - SMILES2Weight
        - FunctionalGroups
        - ArxivLiteratureSearch        (Arxiv2ResultLLM)
        - InspectCSVDataset
        - GenericBayesOpt1D
        - MotifDecomposition
        - VastraVisualise
        - CheckCrystalFile

        We have a COF dataset with pre-cleaned selectivity values:

        COF CSV (descriptors + selectivity, all in one table):
            {CSV_PATH}

        CIF directory for these COFs:
            {cif_dir_str}

        Motif library JSON (MotifPattern schema):
            {motif_lib_str}

        High-level scientific question
        ------------------------------
        We want to know whether any COFs in this dataset have *above-average or at least
        notable* Xe/Kr selectivity, and then to identify the *motifs* that appear to
        drive that high selectivity.

        We also want you to:
        - Visualise selected high-selectivity COFs (full crystals).
        - Visualise 2–3 of the most frequent motifs as isolated coordination
            environments.
        - Store key identifiers (COF IDs, motif names, PNG filenames) in a final
            JSON block so they can be re-used programmatically.

        Error-handling protocol (IMPORTANT)
        -----------------------------------
        Throughout this task, you must be *error-aware* and *error-recoverable*:

        - If ANY tool returns an error-like message (e.g. containing 'Error', 'ERROR',
            'failed', 'ARXIV_TOOL_ERROR', 'Vastra/VESTA visualisation failed', etc.),
            treat that call as FAILED.

        - For a FAILED tool call:
            • You MAY retry the same tool at most ONCE if the failure looks transient
                (e.g. due to a formatting issue you can fix).
            • If the second attempt also fails, DO NOT keep retrying. Instead:
                – Record in your reasoning and in the final summary that this step
                    failed.
                – Skip that specific sub-step and continue with the rest of the
                    pipeline using whatever partial information you do have.

        - Do NOT let a single tool failure abort the whole analysis. Always try to
            complete the pipeline as far as possible with partial results.

        - When using Python_REPL, if a piece of code fails, you may fix the code and
            re-run it, but again avoid infinite loops of retries. Two attempts per
            logical operation is a good maximum.

        - In your final summary, clearly mention any missing pieces or skipped
            steps caused by tool failures.
            
        Meta constraints (NO HALLUCINATIONS)
        ------------------------------------
        - You MUST NOT claim to have used a tool unless there is an actual
        tool invocation ("Action: <ToolName>" followed by an "Observation: ...")
        during this conversation.

        - You MUST NOT invent or guess:
            • motif_counts dicts,
            • motif names,
            • PNG paths,
            • JSON files,
            • or any other file content.
        All such values MUST come from actual tool outputs or from the local
        filesystem via Python_REPL.

        - When using Python_REPL:
            • You MUST load motif_counts from the JSON files actually written by
            MotifDecomposition (the `full_result_path` fields returned in its
            Observations).
            • You MUST NOT type example motif_counts or use comments like
            "to be replaced with actual loaded data".
            Instead, write code that opens those JSON files and extracts
            `result["motif_counts"]`.

        - Every PNG path you report in the final JSON MUST correspond either to:
            • a successful VastraVisualise tool output containing
            "Saved structure image to: <path>", or
            • be set to null if the visualisation failed or was never run.

        - You MUST NOT use placeholder tokens like "<top_motif_name_1>" in any
        tool call or in the final JSON. All motif_name values MUST be actual
        names present in the motif_counts or motif library.


        === TASK STEPS (follow them, using tool calls explicitly) ===

        1) Literature benchmark (ArxivLiteratureSearch)
        ------------------------------------------------
        - Use **ArxivLiteratureSearch** exactly ONCE with a query along the lines of:

            "Xe/Kr selectivity in MOFs or COFs at ambient or near-ambient conditions;
                typical values; what counts as 'high' or 'notable' selectivity?"

        - If the call fails, retry ONCE with a slightly simpler query
            (e.g. "Xe Kr selectivity MOF COF typical high values ambient").

        - From the returned papers, summarise in 2–4 sentences:
            • Typical ranges of Xe/Kr selectivity reported.
            • Rough threshold for what the community would call "notable" or "high"
                selectivity (approximate, not exact).

        - Record in your notes:
            • A scalar or small range for 'notable' selectivity, e.g. 10–20, 20–50, etc.
            • Any caveats from the literature (pressure, temperature, etc.).


        2) CSV introspection (InspectCSVDataset)
        --------------------------------------
        - Use **InspectCSVDataset** exactly ONCE with the following JSON config
            string as input (do NOT modify this string):

            {inspect_config_json}

        - If the call fails, retry ONCE. If still failing, describe what you *would*
            expect (based on assumptions) and continue.

        - From the InspectCSVDataset output, report:
            • The total number of rows and columns.
            • The likely ID-like column(s) (e.g. 'crystal_name').
            • Which numeric columns look like they could be Xe/Kr selectivity
                (e.g. 'selectivity_Xe/Kr', 'selectivity_XeKr', etc.).
            • Any obvious missing-data patterns for the selectivity column you think
                is correct.

        - Decide and clearly state:
            • The *exact* column name you will treat as the Xe/Kr selectivity column.
            • The *exact* column name you will treat as the ID column.

        - Store these two column names for later use in a small internal record
            (and later in the JSON summary).


        3) 1D Bayesian optimisation on selectivity (GenericBayesOpt1D)
        -----------------------------------------------------------
        - Use **GenericBayesOpt1D** exactly ONCE with the following JSON config
            string as input (do NOT modify this string):

            {generic_bo_config_json}

            This means:
            • csv_path = the COF CSV above.
            • goal    = "selectivity" (as a fuzzy label – the tool must resolve it
                to a real column).
            • top_k   = 10.
            • id_column and maximise should be inferred or validated by the tool
                based on the CSV structure.

        - If the call fails because of a simple mis-specification (e.g. cannot find
            the goal column), you may retry ONCE with a modified JSON (e.g. changing
            the 'goal' to the actual column name you identified in step 2). If this
            second call also fails, stop trying and explain why, then continue with
            a simpler "top-10 by value" analysis using Python_REPL instead.

        - After a successful BO call (or your fallback Python_REPL approach), summarise:
            • Which *actual* column name was used as the selectivity objective.
            • Which column was treated as the ID.
            • How many rows had finite (observed) selectivity values vs how many
                were treated as unmeasured candidates (if available).
            • The top 10 suggested COFs (by ID), including for each:
                – the ID,
                – predicted mean ± uncertainty for the selectivity (or raw value),
                – any acquisition / EI score if available.

        - Extract and clearly list the **top 10 IDs** you will use for motif analysis
            in the next step (e.g. as a bullet list).

        - Store these 10 IDs and their numeric scores in a structured list for the
            final JSON summary.

        4) Motif analysis on top-10 COFs (MotifDecomposition + Python_REPL)
        --------------------------------------------------------------------
        - For each of the top 10 IDs from step (3):
            • Construct the CIF path as:
                    {cif_dir_str}\\<ID value>
                assuming the ID column matches the CIF filename (e.g.
                "07000N2_ddec.cif").
            • Use **CheckCrystalFile** (optionally) to verify that the CIF exists.
                If the CIF does not exist or the check fails, skip that COF but
                keep going with the others.
            • Call **MotifDecomposition** at least once per *existing* CIF, with a
                JSON config of the form:

                {
                    "mode": "all",
                    "cif_path": "<FULL_PATH_TO_CIF>",
                    "motif_library_path": "{motif_lib_str}",
                    "allow_overlap": true
                }

                If a decomposition fails for a particular CIF (even after one retry
                with a minimal config), skip that COF for motif statistics but
                continue with the rest.

        - Use **Python_REPL** to:
            • Load each successful motif JSON from the `full_result_path` returned
                by MotifDecomposition.
            • For each COF, construct a *set* of motif names, e.g.
                    motifs_i = { m["motif_name"] for m in data["motifs"] }.
            • Build a frequency dictionary over the top-10 COFs:
                    freq[motif_name] = number of COFs in which that motif appears.

        - Instead of requiring motifs to appear in *all* top-10 COFs (which is
            usually too strict and biases toward trivial backbone motifs like CH1),
            define:

            • high-frequency motifs = motifs that appear in at least
                ceil(alpha * K) COFs, where K is the number of successfully
                processed top-10 COFs and alpha is between 0.4 and 0.6.
                (For K=10, this means motifs seen in ≥4–6 COFs.)

            • optional intersection motifs = motifs present in every successfully
                processed top-10 COF (for diagnostic purposes only).

        - If the global motif corpus is available (e.g. `corpus.jsonl` over all
            COFs), you MAY additionally compute a background frequency for each
            motif:
                freq_bg[m] = fraction of all COFs in the corpus that contain m,
            and rank motifs by an enrichment ratio such as:
                enrichment_ratio[m] = freq_top[m] / max(freq_bg[m], 1e-6).

        - In all cases you MUST:
            • Filter out obviously trivial motifs such as "CH1" and "HC1" unless
                they are explicitly very highly enriched.
            • Select a final list of 3–5 "selectivity-associated" motifs, prioritising:
                    – high frequency in the top-10 COFs, and
                    – strong enrichment relative to the global corpus if available.

        - Store in your internal record (and later JSON summary):
            • For each such motif: motif name/ID, its frequency among the top-10
                COFs, and any enrichment score you computed.
            • The list of motif names you intend to visualise in step (5),
                typically 2–3 motifs.

        -------------------------------------------------
        
        - When aggregating motif frequencies, you MUST:
            • Collect the 'full_result_path' values returned by each successful
                MotifDecomposition call (e.g. "motif_results\\19440N2_ddec_motifs_all.json"),
            • Use Python_REPL to open each of these JSON files from disk, e.g.:

                import json, pathlib
                paths = [
                    "motif_results\\19440N2_ddec_motifs_all.json",
                    "motif_results\\19441N2_ddec_motifs_all.json",
                    ...
                ]

                per_cof_motif_sets = []
                for p in paths:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Each file has a list under "motifs"; each entry has "motif_name"
                    names = {m["motif_name"] for m in data["motifs"]}
                    per_cof_motif_sets.append(names)

                # Build frequency dict: in how many COFs does each motif appear?
                freq = {}
                for names in per_cof_motif_sets:
                    for name in names:
                        freq[name] = freq.get(name, 0) + 1

            • Then compute presence/frequency from freq.


        - You MUST NOT invent motif_counts by hand or write "example" or
            placeholder motif_counts. All counts MUST come from these JSON files. 


        5) Visualise best COFs and top motifs (VastraVisualise, per-call filenames)
        ------------------------------------------------------------------------
        HARD REQUIREMENTS FOR THIS STEP:
            - You MUST actually call VastraVisualise for:
                • at least 1 high-selectivity COF, and
                • at least 2 motifs from your "top motifs" list.
            - You MUST NOT simply describe what you would visualise.
            - You MUST record the exact "Saved structure image to: ..." strings
            from VastraVisualise Observations and reuse those paths in the
            final JSON.

        5.1 Choose 1–3 COFs from the analysed subset of top-10 that:
            • have high selectivity,
            • and share many of the high-frequency motifs you identified.

        5.2 For each chosen COF, build a per-call JSON config for
            **VastraVisualise** in CIF mode with a distinct 'output_png'.
            Example (you MUST adapt <ID>):

                {{
                "mode": "cif",
                "cif_path": "C:\\Users\\brown\\Documents\\PhD\\Winter School\\COF_crystals\\crystals\\<ID>",
                "output_png": "viz/selectivity_cof_<ID_NO_EXT>.png"
                }}

            where <ID_NO_EXT> is the ID without ".cif" and any path separators
            replaced by underscores.

            - You MUST call VastraVisualise with this JSON string as input.
            - If a VastraVisualise call fails (even after one retry), record
                the failure in your notes and set visualisation_png=null for that COF.

        5.3 Visualise 2–3 of the most frequent motifs themselves using
            **VastraVisualise** in motif mode with per-call filenames. For each
            motif name (e.g. "CC2"), call VastraVisualise with a JSON string:

                {{
                "mode": "motif",
                "motif_library_path": "chemcrow\\tools\\New\\motifs_cof_from_library.json",
                "motif_name": "<MOTIF_NAME>",
                "geometry": "auto",
                "output_png": "viz/motif_<MOTIF_NAME>.png"
                }}

            - <MOTIF_NAME> MUST be an actual motif name that appears in your
                motif_counts across the top-10 COFs.
            - Do NOT use placeholders like "<top_motif_name_1>".

        5.4 For every successful VastraVisualise call, you MUST:
            • capture the "Saved structure image to: ..." message in your
                notes, and
            • propagate the exact PNG path into the final JSON summary.

            If VastraVisualise returns any error-like text, treat that call as
            FAILED, mention it in "notes", and set visualisation_png=null for
            that item.



        6) Final summary (human + machine-readable JSON)
        ----------------------------------------------
        - Provide a succinct but technically detailed final *human-readable* summary
            that includes:
            • The literature benchmark for "notable" Xe/Kr selectivity.
            • The selectivity range of the top-10 COFs from our dataset and how they
                compare to that benchmark.
            • The key motifs you identified as most associated with high selectivity
                (intersection + high-frequency motifs).
            • Which specific COFs (IDs) exemplify these motifs and high selectivity.
            • Which motifs and COFs you visualised with VastraVisualise and how
                their local environments look qualitatively (e.g. octahedral, trigonal
                planar, etc.), based on the rendered images.
            • Any caveats (e.g. limited sample size, descriptor biases, BO model
                uncertainty, any failed tool calls).

        - THEN produce a **machine-readable JSON block** that summarises the key
            identifiers and filenames. The JSON must be printed as the ONLY content
            inside a fenced code block labelled exactly 'JSON_SUMMARY', like:

            ```JSON_SUMMARY
            {{ ...json here... }}
            ```

            The JSON object should have at least the following structure:

            {{
                "selectivity_column": "<name of selectivity column>",
                "id_column": "<name of ID column>",
                "literature_notable_range": "<short text or small numeric range>",
                "top_cofs": [
                {{
                    "id": "<COF_ID>",
                    "selectivity_value": <numeric or null>,
                    "selectivity_pred_mean": <numeric or null>,
                    "selectivity_pred_std": <numeric or null>,
                    "acquisition_score": <numeric or null>,
                    "visualisation_png": "<path or null>"
                }},
                ...
                ],
                "motifs": [
                {{
                    "motif_name": "<motif ID / name>",
                    "frequency_in_top_cofs": <integer>,
                    "visualisation_png": "<path or null>"
                }},
                ...
                ],
                "notes": [
                "Any important caveats, including tool failures, as short strings."
                ]
            }}

            - Use null where a value is unknown or not applicable.
            - Ensure the JSON is syntactically valid.
            - Include the actual PNG paths you used in VastraVisualise calls wherever possible.

        IMPORTANT:
        - Follow the steps in order.
        - Use the named tools explicitly as indicated.
        - Be error-aware, retry at most once where sensible, and continue with
            partial results when something fails.
        - Prefer Python_REPL for any non-trivial parsing/aggregation work.
            """
    logger.info("\n=== TEST PROMPT (robust selectivity + motif pipeline) ===\n")
    logger.info(test_prompt)
    logger.info("\n=== MODEL RESPONSE ===\n")

    try:
        answer = chem_model.run(test_prompt)
        # Log and also echo the answer
        logger.info(answer)
    except Exception as e:
        # Top-level safety: do not crash the test runner without a message.
        logger.error("[TOP-LEVEL ERROR] chem_model.run raised an exception:")
        logger.exception(e)


if __name__ == "__main__":
    main()

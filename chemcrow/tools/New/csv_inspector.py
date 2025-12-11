import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain.tools import BaseTool


class InspectCSVDataset(BaseTool):
    """
    ChemCrow tool to introspect one descriptor CSV and one or more property
    CSVs, primarily to help configure COFMultiObjectiveBO or similar BO / ML
    tools.

    INPUT (string): MUST be a JSON object with keys like:

        {
          "descriptor_csv": "C:/.../cof_descriptors.csv",
          "property_csvs": ["C:/.../gcmc_calculations.csv"],
          "cif_dir": "C:/.../crystals"        # optional
        }

    What it does:

      - Loads the descriptor CSV and each property CSV.
      - Reports for each CSV:
          * path, exists?,
          * number of rows and columns,
          * list of columns with dtypes.
      - Computes columns common across ALL CSVs.
      - Heuristically identifies candidate ID columns:
          * columns common to all tables,
          * typically string-like or categorical,
          * with a high proportion of unique values.
      - Identifies candidate numeric "property" columns in the property CSVs:
          * numeric columns that are not obviously ID-like.
      - If cif_dir is given and exists, reports how many CIFs are present and
        whether any common ID columns look like they match the CIF basenames.

    This is meant to be used before configuring COFMultiObjectiveBO, so that
    the agent (or human) can pick a consistent id_column and reasonable target
    properties.
    """

    name = "InspectCSVDataset"
    description = (
        "Inspect one descriptor CSV and one or more property CSVs, to list "
        "their columns, dtypes, common columns, candidate ID columns, and "
        "candidate numeric properties. Input MUST be a JSON string with at "
        "least 'descriptor_csv' and 'property_csvs' keys. Use this before "
        "calling COFMultiObjectiveBO to decide on 'id_column' and "
        "'target_properties'."
    )

    def _run(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return (
                "InspectCSVDataset: no input provided. Expected a JSON string with "
                "'descriptor_csv' and 'property_csvs'."
            )

        try:
            cfg = json.loads(query)
        except json.JSONDecodeError as e:
            return f"InspectCSVDataset: failed to parse JSON input: {e}"

        descriptor_csv = cfg.get("descriptor_csv")
        property_csvs = cfg.get("property_csvs")
        cif_dir = cfg.get("cif_dir", "")

        if not descriptor_csv or not property_csvs:
            return (
                "InspectCSVDataset: config must include 'descriptor_csv' and "
                "'property_csvs' (list of CSV paths)."
            )

        if isinstance(property_csvs, str):
            property_csvs = [property_csvs]

        # Load CSVs
        try:
            desc_df = self._load_csv(descriptor_csv)
        except Exception as e:
            return f"InspectCSVDataset: error loading descriptor_csv '{descriptor_csv}': {e}"

        prop_dfs = []
        for p in property_csvs:
            try:
                df = self._load_csv(p)
                prop_dfs.append((p, df))
            except Exception as e:
                return f"InspectCSVDataset: error loading property_csv '{p}': {e}"

        lines: List[str] = []

        # --- Descriptor CSV summary ---
        lines.append("=== DESCRIPTOR CSV ===")
        lines.append(f"Path: {descriptor_csv}")
        lines.append(f"Exists on disk? {'YES' if os.path.isfile(descriptor_csv) else 'NO'}")
        lines.append(f"Shape: {desc_df.shape[0]} rows × {desc_df.shape[1]} columns")
        lines.append("Columns (name : dtype):")
        for c in desc_df.columns:
            lines.append(f"  - {c} : {desc_df[c].dtype}")
        lines.append("")

        # --- Property CSVs summary ---
        lines.append("=== PROPERTY CSVs ===")
        for path, df in prop_dfs:
            lines.append(f"Path: {path}")
            lines.append(f"  Exists on disk? {'YES' if os.path.isfile(path) else 'NO'}")
            lines.append(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            lines.append("  Columns (name : dtype):")
            for c in df.columns:
                lines.append(f"    - {c} : {df[c].dtype}")
            lines.append("")

        # --- Common columns & candidate ID columns ---
        all_dfs = [desc_df] + [df for _, df in prop_dfs]
        common_cols = set(all_dfs[0].columns)
        for df in all_dfs[1:]:
            common_cols &= set(df.columns)

        lines.append("=== COMMON COLUMNS ACROSS ALL CSVs ===")
        if not common_cols:
            lines.append("None.")
        else:
            for c in sorted(common_cols):
                dtypes = [str(df[c].dtype) for df in all_dfs]
                lines.append(f"  - {c}  (dtypes: {', '.join(dtypes)})")
        lines.append("")

        candidate_ids = self._find_candidate_id_columns(all_dfs, list(common_cols))
        lines.append("=== CANDIDATE ID COLUMNS ===")
        if not candidate_ids:
            lines.append(
                "No strong ID candidates detected, but common string-like columns "
                "are still candidates if they exist."
            )
        else:
            for c, uniq_frac, example_vals in candidate_ids:
                ex_str = ", ".join(example_vals)
                lines.append(
                    f"  - {c}  (unique_ratio≈{uniq_frac:.3f}; examples: {ex_str})"
                )
        lines.append("")

        # --- Candidate numeric property columns ---
        lines.append("=== CANDIDATE NUMERIC TARGET PROPERTIES (from property CSVs) ===")
        numeric_counts: dict[str, int] = {}
        for _, df in prop_dfs:
            for c in df.columns:
                if np.issubdtype(df[c].dtype, np.number):
                    numeric_counts[c] = numeric_counts.get(c, 0) + 1

        # Only keep numeric columns that appear in *all* property CSVs
        n_props = len(prop_dfs)
        candidate_numeric = [
            c for c, cnt in numeric_counts.items() if cnt == n_props
        ]

        # Exclude obvious IDs from numeric candidates
        id_like_names = {"id", "idx", "index"}
        id_candidate_names = {c for c, _, _ in candidate_ids}
        filtered_numeric = [
            c for c in candidate_numeric
            if c.lower() not in id_like_names and c not in id_candidate_names
        ]

        if not filtered_numeric:
            lines.append(
                "No strong candidate numeric properties found that are common to all "
                "property CSVs and not ID-like."
            )
        else:
            for c in sorted(filtered_numeric):
                col_dtypes = [str(df[c].dtype) for _, df in prop_dfs if c in df.columns]
                lines.append(f"  - {c}  (dtypes: {', '.join(col_dtypes)})")
        lines.append("")

        # --- CIF dir check (optional) ---
        if cif_dir and os.path.isdir(cif_dir):
            lines.append("=== CIF DIRECTORY CHECK ===")
            try:
                cif_basenames = {
                    os.path.splitext(fn)[0]
                    for fn in os.listdir(cif_dir)
                    if fn.lower().endswith(".cif")
                }
            except Exception as e:
                lines.append(f"Error listing CIFs in '{cif_dir}': {e}")
                cif_basenames = set()

            lines.append(f"CIF directory: {cif_dir}")
            lines.append(f"Number of CIF files found: {len(cif_basenames)}")

            # For each candidate ID, see how many values match CIF basenames
            if candidate_ids and cif_basenames:
                for c, _, _ in candidate_ids:
                    vals = set(desc_df[c].astype(str).values)
                    overlap = len(vals & cif_basenames)
                    lines.append(
                        f"  - Candidate ID '{c}': {overlap} values match CIF basenames."
                    )
            lines.append("")

        return "\n".join(lines)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async.")

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _find_candidate_id_columns(
        dfs: List[pd.DataFrame],
        common_cols: List[str],
        min_unique_frac: float = 0.3,
        max_unique_frac: float = 0.99,
    ):
        """
        Heuristic: ID-like columns are common across all tables, string-like or
        low-cardinality, with a decent fraction of unique values.
        Returns list of tuples (col_name, unique_ratio, example_values).
        """
        candidates = []
        if not common_cols:
            return candidates

        base_df = dfs[0]

        for c in common_cols:
            series = base_df[c]
            n = len(series)
            if n == 0:
                continue

            # Heuristic: string-like or "object" dtype are most likely IDs
            if series.dtype == object or series.dtype == "string":
                unique_vals = series.astype(str).dropna().unique()
                uniq_ratio = len(unique_vals) / max(n, 1)
                if min_unique_frac <= uniq_ratio <= max_unique_frac:
                    examples = [str(v) for v in unique_vals[:5]]
                    candidates.append((c, uniq_ratio, examples))
            else:
                # Also consider numerical columns that look like indices
                unique_vals = series.dropna().unique()
                uniq_ratio = len(unique_vals) / max(n, 1)
                col_lower = c.lower()
                is_indexy_name = any(k in col_lower for k in ("id", "idx", "index"))
                if is_indexy_name and min_unique_frac <= uniq_ratio <= 1.0:
                    examples = [str(v) for v in unique_vals[:5]]
                    candidates.append((c, uniq_ratio, examples))

        return candidates

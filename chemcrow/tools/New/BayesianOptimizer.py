import os
import json
import glob
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel as C,
)
from sklearn.preprocessing import StandardScaler


# ===========================
#  Utility: normal PDF / CDF
# ===========================

def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """
    Approximate error function (vectorised A&S 7.1.26).
    Good enough for EI in BO.
    """
    # handle vector input
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x_abs = np.abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (
        (((a5 * t + a4) * t + a3) * t + a2) * t + a1
    ) * t * np.exp(-x_abs * x_abs)

    return sign * y


# ===========================
#  CSV / ID handling
# ===========================

def _infer_id_column(dfs: List[pd.DataFrame]) -> str:
    """
    Infer a crystal identifier column present across all dataframes.
    Prioritises typical names like 'crystal_name' or 'xtal'.
    """
    if not dfs:
        raise ValueError("No dataframes provided to infer ID column")

    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    preferred = ["crystal_name", "xtal", "name", "id"]

    for col in preferred:
        if col in common:
            return col

    if not common:
        # Fall back to the first column of the first dataframe
        return dfs[0].columns[0]

    # Prefer string-like columns among the common set
    for col in dfs[0].columns:
        if col in common and dfs[0][col].dtype == object:
            return col

    # Otherwise just take an arbitrary common column
    return list(common)[0]


def _load_descriptor_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Descriptor CSV not found: {path}")
    return pd.read_csv(path)


def _load_property_csvs(paths: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Property CSV not found: {p}")
        dfs.append(pd.read_csv(p))
    return dfs


def _build_merged_table(
    cif_dir: str,
    descriptor_csv: str,
    property_csvs: List[str],
    target_properties: List[str],
    id_column: Optional[str] = None,
    property_agg: str = "mean",
) -> pd.DataFrame:
    """
    Load descriptors + property CSV(s), join on the ID column,
    and (optionally) restrict to crystals for which we have CIFs.

    target_properties: list of columns (objectives) we want to maximise.
    """
    desc_df = _load_descriptor_csv(descriptor_csv)
    prop_dfs = _load_property_csvs(property_csvs)

    # Infer ID column if needed
    if id_column is None:
        id_column = _infer_id_column([desc_df] + prop_dfs)

    if id_column not in desc_df.columns:
        raise ValueError(
            f"Inferred ID column '{id_column}' not found in descriptor CSV."
        )

    # Merge all property CSVs, aggregating numeric columns by ID
    merged_props = None
    for df in prop_dfs:
        if id_column not in df.columns:
            raise ValueError(
                f"ID column '{id_column}' not found in property CSV with columns {list(df.columns)}"
            )

        if property_agg == "max":
            g = df.groupby(id_column).max(numeric_only=True)
        elif property_agg == "min":
            g = df.groupby(id_column).min(numeric_only=True)
        else:
            g = df.groupby(id_column).mean(numeric_only=True)

        g = g.reset_index()

        if merged_props is None:
            merged_props = g
        else:
            merged_props = pd.merge(
                merged_props,
                g,
                on=id_column,
                how="outer",
                suffixes=("", "_prop"),
            )

    if merged_props is None:
        raise ValueError("No property data could be loaded / merged.")

    merged = pd.merge(desc_df, merged_props, on=id_column, how="inner")

    # Restrict to COFs we have CIFs for (if directory given)
    if cif_dir and os.path.isdir(cif_dir):
        cif_files = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(cif_dir, "*.cif"))
        }
        if cif_files:
            mask = merged[id_column].astype(str).isin(cif_files)
            merged = merged[mask].copy()

    # Check all target properties exist
    missing = [t for t in target_properties if t not in merged.columns]
    if missing:
        raise ValueError(
            f"Target properties {missing} not found in merged data. "
            f"Available columns: {list(merged.columns)}"
        )

    merged.attrs["id_column"] = id_column
    return merged


# ===========================
#  Pareto utilities
# ===========================

def _compute_pareto_front(
    Y: np.ndarray, maximise: bool = True
) -> np.ndarray:
    """
    Compute indices of Pareto-optimal points in Y (n x m).

    maximise=True means each objective is to be maximised.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    is_dominated = np.zeros(n, dtype=bool)

    # For maximisation, domination means >= in all, > in at least one
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if j == i or is_dominated[j]:
                continue
            if maximise:
                if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                    is_dominated[i] = True
                    break
            else:
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    is_dominated[i] = True
                    break
    return np.where(~is_dominated)[0]


# ===========================
#  Multi-objective BO core
# ===========================

def _fit_multi_gp(
    X_obs: np.ndarray,
    Y_obs: np.ndarray,
) -> Tuple[StandardScaler, List[GaussianProcessRegressor]]:
    """
    Fit one GP per objective on observed points.
    Returns feature scaler + list of fitted GPs.
    """
    scaler = StandardScaler()
    X_obs_scaled = scaler.fit_transform(X_obs)

    gps: List[GaussianProcessRegressor] = []
    for j in range(Y_obs.shape[1]):
        y_j = Y_obs[:, j]

        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) \
                 + WhiteKernel(noise_level=1e-3)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=0,
        )
        gp.fit(X_obs_scaled, y_j)
        gps.append(gp)

    return scaler, gps


def _predict_multi_gp(
    scaler: StandardScaler,
    gps: List[GaussianProcessRegressor],
    X_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict means and stds for all objectives at all points.

    Returns:
      mu_all: (n_points, n_obj)
      sigma_all: (n_points, n_obj)
    """
    X_all_scaled = scaler.transform(X_all)
    n_points = X_all.shape[0]
    n_obj = len(gps)

    mu_all = np.zeros((n_points, n_obj), dtype=float)
    sigma_all = np.zeros((n_points, n_obj), dtype=float)

    for j, gp in enumerate(gps):
        mu_j, sigma_j = gp.predict(X_all_scaled, return_std=True)
        mu_all[:, j] = mu_j
        sigma_all[:, j] = sigma_j

    return mu_all, sigma_all


def _sample_weights(
    n_obj: int,
    weights_cfg: Optional[Dict],
) -> np.ndarray:
    """
    Return an array of weight vectors on the simplex.

    weights_cfg can contain:
      - 'weights': list[float] (single weight vector) OR
                   list[list[float]] (multiple)
      - 'n_weight_samples': int
    """
    if weights_cfg is None:
        weights_cfg = {}

    user_w = weights_cfg.get("weights")
    n_samples = int(weights_cfg.get("n_weight_samples", 4))

    if user_w is not None:
        # Single vector: [w1, w2, ...]
        if isinstance(user_w, list) and user_w and isinstance(user_w[0], (float, int)):
            w = np.asarray(user_w, dtype=float)
            if len(w) != n_obj:
                raise ValueError(
                    f"weights length {len(w)} != number of objectives {n_obj}"
                )
            w = w / w.sum()
            return np.atleast_2d(w)

        # List of vectors: [[...], [...], ...]
        if isinstance(user_w, list) and user_w and isinstance(user_w[0], list):
            W = []
            for vec in user_w:
                v = np.asarray(vec, dtype=float)
                if len(v) != n_obj:
                    raise ValueError(
                        f"One weight vector has length {len(v)} != n_obj {n_obj}"
                    )
                v = v / v.sum()
                W.append(v)
            return np.asarray(W)

        raise ValueError("Invalid 'weights' format in config.")

    # Default: random Dirichlet samples on simplex
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(n_obj, dtype=float), size=n_samples)
    return W


def _run_multiobjective_bo(
    merged: pd.DataFrame,
    target_properties: List[str],
    top_k: int = 15,
    weights_cfg: Optional[Dict] = None,
) -> str:
    """
    Multi-objective BO over a discrete set of COFs.

    - Fits a GP surrogate per objective on points where all objectives are observed.
    - Normalises objectives (z-score).
    - Samples weight vectors on the simplex and forms a linear scalarisation
      g(x) = sum_j w_j * z_j(x) (scalarized normalised objective).
    - Uses Expected Improvement (EI) on g(x) to rank unobserved points,
      aggregating by max EI across sampled weight vectors.

    Returns a human-readable summary string.
    """
    id_column = merged.attrs.get("id_column", merged.columns[0])
    ids = merged[id_column].astype(str).values

    # Numeric features + objectives
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    for t in target_properties:
        if t not in numeric_cols:
            raise ValueError(f"Target property '{t}' is not numeric.")

    feature_cols = [c for c in numeric_cols if c not in target_properties]
    if not feature_cols:
        raise ValueError("No numeric descriptor features found for BO.")

    X_all = merged[feature_cols].values.astype(float)
    Y_all = merged[target_properties].values.astype(float)

    # Observed if all objectives are present
    mask_obs = np.all(np.isfinite(Y_all), axis=1)
    if mask_obs.sum() < max(3, len(target_properties) + 1):
        # Not enough data for proper GP; just Pareto + ranking
        Y_obs = Y_all[mask_obs]
        ids_obs = ids[mask_obs]
        pf_idx = _compute_pareto_front(Y_obs, maximise=True)

        lines = []
        lines.append(
            f"Not enough data for GP-based multi-objective BO "
            f"(only {mask_obs.sum()} fully-observed points; need ≥ {max(3, len(target_properties)+1)})."
        )
        lines.append("")
        lines.append("Pareto-optimal COFs among observed points:")
        for i in pf_idx:
            prop_str = ", ".join(
                f"{name} = {val:.4g}"
                for name, val in zip(target_properties, Y_obs[i])
            )
            lines.append(f"- {ids_obs[i]}: {prop_str}")
        return "\n".join(lines)

    X_obs = X_all[mask_obs]
    Y_obs = Y_all[mask_obs]
    ids_obs = ids[mask_obs]

    # Fit one GP per objective
    scaler, gps = _fit_multi_gp(X_obs, Y_obs)

    # Predict for all points
    mu_all, sigma_all = _predict_multi_gp(scaler, gps, X_all)

    # Normalise objectives using observed Y
    y_mean = Y_obs.mean(axis=0)
    y_std = Y_obs.std(axis=0)
    y_std[y_std == 0.0] = 1.0

    # Predicted mean & std in z-space
    z_mu_all = (mu_all - y_mean) / y_std
    z_sigma_all = sigma_all / y_std

    # Observed z-values (true) for computing g_best
    Z_obs = (Y_obs - y_mean) / y_std

    n_points, n_obj = z_mu_all.shape

    # Candidate indices: prioritise unobserved
    unobs_mask = ~mask_obs
    if unobs_mask.any():
        candidate_idx = np.where(unobs_mask)[0]
    else:
        candidate_idx = np.arange(n_points)

    # Sample weights on simplex
    W = _sample_weights(n_obj, weights_cfg)
    n_w = W.shape[0]

    # For each weight vector, compute EI on scalarised objective
    EI_mat = np.zeros((n_w, candidate_idx.shape[0]), dtype=float)

    xi = 0.01  # exploration parameter

    for w_i, w in enumerate(W):
        # scalarised normalised means at all points
        g_mu_all = (z_mu_all * w).sum(axis=1)
        g_sigma_all = np.sqrt(
            np.sum((z_sigma_all * w) ** 2, axis=1)
        )

        # Best scalarised value over observed points
        g_obs = (Z_obs * w).sum(axis=1)
        g_best = float(np.max(g_obs))

        # EI for candidates
        mu_c = g_mu_all[candidate_idx]
        sig_c = g_sigma_all[candidate_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu_c - g_best - xi) / sig_c
            cdf_z = _norm_cdf(z)
            pdf_z = _norm_pdf(z)
            ei = (mu_c - g_best - xi) * cdf_z + sig_c * pdf_z
            ei[sig_c <= 0.0] = 0.0

        EI_mat[w_i, :] = ei

    # Aggregate EI across weights: max EI per candidate
    ei_max = EI_mat.max(axis=0)
    # Sort candidates by EI descending
    ranking_local = np.argsort(-ei_max)
    top_k = min(top_k, len(ranking_local))
    best_local = ranking_local[:top_k]
    best_idx = candidate_idx[best_local]

    # Pareto front among observed points (based on true Y_obs)
    pf_idx = _compute_pareto_front(Y_obs, maximise=True)

    # Build summary
    lines = []
    lines.append(
        f"Multi-objective Bayesian Optimization over {n_points} COFs "
        f"with {n_obj} objectives: {', '.join(target_properties)} (all maximised)."
    )
    lines.append(
        f"Fully-observed points: {len(Y_obs)}; unobserved candidates: {int(unobs_mask.sum())}."
    )
    lines.append(
        f"Using scalarisation-based MOBO with {n_w} weight vector(s) on the objective simplex "
        f"and EI as acquisition."
    )
    lines.append("")

    # Current Pareto front
    lines.append("Pareto-optimal COFs among currently observed points:")
    for i in pf_idx:
        prop_str = ", ".join(
            f"{name} = {val:.4g}"
            for name, val in zip(target_properties, Y_obs[i])
        )
        lines.append(f"- {ids_obs[i]}: {prop_str}")

    # BO suggestions
    lines.append("")
    lines.append(f"Top {top_k} BO suggestions (max EI over sampled weights):")
    for idx, ei_val in zip(best_idx, ei_max[best_local]):
        # predicted objectives for this candidate
        pred_props = [
            f"{name} ≈ {mu_all[idx, j]:.4g} ± {sigma_all[idx, j]:.4g}"
            for j, name in enumerate(target_properties)
        ]
        lines.append(
            f"- {ids[idx]}: EI ≈ {ei_val:.4g}; "
            + "; ".join(pred_props)
        )

    lines.append("")
    lines.append(
        "Note: suggestions are generated via linear scalarisation of "
        "normalised objectives plus Expected Improvement, which is a standard "
        "scalarisation-based multi-objective BO strategy."
    )

    return "\n".join(lines)


def run_cof_multi_bo_from_config(config_json: str) -> str:
    """
    Entry point: takes a JSON string describing the multi-objective BO problem
    and returns a text summary.

    Expected JSON structure:

    {
      "cif_dir": "C:/.../crystals",                # optional
      "descriptor_csv": "C:/.../cof_descriptors.csv",
      "property_csvs": ["C:/.../gcmc_calculations.csv", "..."],
      "target_properties": [
          "⟨N⟩ (mmol/g)",
          "selectivity Xe/Kr"
      ],
      "id_column": "crystal_name",                 # optional; inferred if omitted
      "property_agg": "mean",                      # optional: "mean" | "max" | "min"
      "top_k": 15,                                 # optional
      "weights": [0.5, 0.5],                       # optional: or [[...], [...], ...]
      "n_weight_samples": 4                        # optional if no weights given
    }

    All objectives are assumed to be maximised.
    """
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as e:
        return f"Failed to parse JSON config: {e}"

    descriptor_csv = cfg.get("descriptor_csv")
    property_csvs = cfg.get("property_csvs")
    target_properties = cfg.get("target_properties")

    if not descriptor_csv or not property_csvs or not target_properties:
        return (
            "Config must include 'descriptor_csv', 'property_csvs', "
            "and 'target_properties' (list of columns to maximise)."
        )

    if isinstance(property_csvs, str):
        property_csvs = [property_csvs]

    if not isinstance(target_properties, list) or len(target_properties) < 2:
        return (
            "'target_properties' must be a list of at least two objective "
            "column names for multi-objective BO."
        )

    cif_dir = cfg.get("cif_dir", "")
    id_column = cfg.get("id_column")
    property_agg = cfg.get("property_agg", "mean")
    top_k = int(cfg.get("top_k", 15))

    weights_cfg = {
        k: v for k, v in cfg.items()
        if k in ("weights", "n_weight_samples")
    } or None

    try:
        merged = _build_merged_table(
            cif_dir=cif_dir,
            descriptor_csv=descriptor_csv,
            property_csvs=property_csvs,
            target_properties=target_properties,
            id_column=id_column,
            property_agg=property_agg,
        )
        summary = _run_multiobjective_bo(
            merged=merged,
            target_properties=target_properties,
            top_k=top_k,
            weights_cfg=weights_cfg,
        )
        return summary
    except Exception as e:
        return f"Error during multi-objective COF BO: {e}"


# ===========================
#  ChemCrow tool wrapper
# ===========================

class COFMultiObjectiveBO(BaseTool):
    """
    ChemCrow tool for multi-objective Bayesian optimisation over a finite COF
    library using precomputed descriptors and property tables.

    Expected data:
      - A descriptor CSV: one row per crystal, numeric descriptor columns, and
        an identifier column (e.g. 'crystal_name').
      - One or more property CSVs: measured or simulated properties per crystal,
        joined by the same identifier (e.g. uptake, selectivity, band gap).
      - A list of target_properties: the columns you want to MAXIMISE jointly.

    What the tool does:
      - Merges descriptors and property CSVs on the crystal ID (optionally
        restricting to crystals that actually have CIFs in 'cif_dir').
      - Identifies points where all target_properties are observed.
      - Fits one Gaussian Process regressor per objective on those points.
      - Normalises the objectives and runs scalarisation-based multi-objective
        BO: samples weight vectors on the simplex, forms linear combos of the
        normalised objectives, and applies Expected Improvement (EI).
      - Ranks unobserved COFs by EI and reports the top candidates, with
        predicted mean ± uncertainty for each objective.
      - Also computes and reports the current Pareto front among observed
        points, giving you a snapshot of the best trade-offs realised so far.

    Why this is useful:
      - It turns a static COF property table into a decision-support tool:
        instead of scanning thousands of rows manually, you get a ranked list
        of promising candidates that balance multiple objectives (e.g. uptake
        vs selectivity, stability vs performance).
      - It respects the discrete nature of your library — it never proposes
        fictitious structures, only re-orders the ones you already have.
      - The Pareto front output is valuable even without BO: it tells you
        which existing COFs are non-dominated with respect to your objectives.

    Limitations / assumptions:
      - Requires at least a handful of fully-observed points (all objectives
        present) to fit the GPs; otherwise it falls back to just reporting the
        Pareto front among observed points.
      - All objectives are assumed to be maximised. If you care about minimising
        a quantity, you should pre-transform it (e.g. use -property).
      - The quality of suggestions depends on descriptor quality, property noise,
        and the assumption that GPs are a reasonable surrogate for the mapping
        from descriptors to properties.
    """

    name = "COFMultiObjectiveBO"
    description = (
        "Run multi-objective Bayesian optimisation over a COF dataset defined by "
        "a descriptor CSV and one or more property CSVs. Input MUST be a JSON "
        "string describing paths and columns, including 'descriptor_csv', "
        "'property_csvs', and a list of 'target_properties' (objectives to "
        "MAXIMISE). The tool:\n"
        "  • merges descriptors and properties on a crystal ID,\n"
        "  • fits a Gaussian Process surrogate for each objective,\n"
        "  • uses scalarisation + Expected Improvement to rank untested COFs,\n"
        "  • reports the current Pareto front and the top BO suggestions.\n"
        "Use this when you want to systematically prioritise which COFs to "
        "synthesise or simulate next given multiple competing objectives "
        "(e.g. uptake and selectivity) over a finite library. It outputs a "
        "human-readable text summary rather than raw arrays, so it can be used "
        "directly in an LLM agent loop."
    )

    llm: Optional[BaseLanguageModel] = None  # not used, kept for symmetry


    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__()
        self.llm = llm

    def _run(self, query: str) -> str:
        return run_cof_multi_bo_from_config(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async.")

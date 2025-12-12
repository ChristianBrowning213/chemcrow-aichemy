import os
import json
import math
import re
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from difflib import get_close_matches
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


# ===========================
#  Utility: normal PDF / CDF
# ===========================

def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    """
    Approximate error function (vectorised A&S 7.1.26).
    Good enough for EI/PI.
    """
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


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


# ===========================
#  CSV / ID / goal handling
# ===========================

def _infer_id_column_single(df: pd.DataFrame) -> str:
    preferred = ["crystal_name", "xtal", "id", "name", "identifier"]
    for col in preferred:
        if col in df.columns:
            return col

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        return obj_cols[0]
    return df.columns[0]


def _normalise_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _resolve_objective_column(
    df: pd.DataFrame,
    goal: Optional[str],
    explicit: Optional[str] = None,
) -> str:
    cols = list(df.columns)

    if explicit:
        if explicit in cols:
            return explicit
        for c in cols:
            if c.lower() == explicit.lower():
                return c
        raise ValueError(
            f"Explicit target_column '{explicit}' not found. Columns: {cols}"
        )

    if goal is None:
        raise ValueError(
            "Config must include either 'target_column' or free-text 'goal'."
        )

    if goal in cols:
        return goal
    for c in cols:
        if c.lower() == goal.lower():
            return c

    norm_goal = _normalise_label(goal)
    norm_map = {c: _normalise_label(c) for c in cols}

    for c, n in norm_map.items():
        if n == norm_goal:
            return c

    candidates = [c for c, n in norm_map.items() if norm_goal in n]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        def score(col: str) -> int:
            return len(set(norm_goal) & set(norm_map[col]))
        candidates.sort(key=score, reverse=True)
        return candidates[0]

    close = get_close_matches(goal, cols, n=1, cutoff=0.6)
    if close:
        return close[0]

    raise ValueError(
        f"Could not resolve goal '{goal}' to any column. Columns: {cols}"
    )


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _impute_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].astype(float)
    X_imp = X.copy()
    for c in feature_cols:
        col = X_imp[c]
        if col.isnull().any():
            med = float(col.median()) if not math.isnan(col.median()) else 0.0
            X_imp[c] = col.fillna(med)
    return X_imp.values


# ===========================
#  Acquisition functions
# ===========================

def _acq_ei(mu: np.ndarray, sigma: np.ndarray, y_best: float, maximise: bool, xi: float) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)

    sign = 1.0 if maximise else -1.0
    sigma_safe = np.maximum(sigma, 1e-12)

    z = (sign * (mu - y_best) - xi) / sigma_safe
    cdf_z = _norm_cdf(z)
    pdf_z = _norm_pdf(z)
    ei = sign * (mu - y_best - xi) * cdf_z + sigma_safe * pdf_z
    ei[sigma <= 0.0] = 0.0
    return ei


def _acq_pi(mu: np.ndarray, sigma: np.ndarray, y_best: float, maximise: bool, xi: float) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sign = 1.0 if maximise else -1.0
    sigma_safe = np.maximum(sigma, 1e-12)

    z = (sign * (mu - y_best) - xi) / sigma_safe
    pi = _norm_cdf(z)
    pi[sigma <= 0.0] = 0.0
    return pi


def _acq_ucb(mu: np.ndarray, sigma: np.ndarray, maximise: bool, kappa: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sign = 1.0 if maximise else -1.0
    return sign * mu + kappa * sigma


def _acq_thompson(mu: np.ndarray, sigma: np.ndarray, maximise: bool, rng: np.random.Generator) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sample = mu + sigma * rng.standard_normal(size=mu.shape)
    sign = 1.0 if maximise else -1.0
    return sign * sample


# ===========================
#  FULL Offline BO loop
# ===========================

def _run_full_offline_bo(
    df: pd.DataFrame,
    *,
    id_column: Optional[str],
    target_column: str,
    maximise: bool = True,
    top_k: int = 10,
    n_init: int = 8,
    n_iter: int = 30,
    acquisition: str = "ei",      # "ei" | "ucb" | "pi" | "ts"
    xi: float = 0.01,             # for EI/PI
    kappa: float = 2.0,           # for UCB
    seed: int = 0,
    withhold_fraction: float = 0.0,  # if >0, simulate unknowns by hiding this fraction of finite targets
    withhold_ids: Optional[List[str]] = None,  # explicit IDs to hide (overrides fraction)
    save_trace_csv: Optional[str] = None,
) -> str:
    # Resolve ID column
    if id_column is None:
        id_column = _infer_id_column_single(df)
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Columns: {list(df.columns)}")

    ids_all = df[id_column].astype(str).values

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_cols:
        raise ValueError(f"Target '{target_column}' is not numeric. Numeric cols: {numeric_cols}")

    feature_cols = [c for c in numeric_cols if c != target_column]
    if not feature_cols:
        raise ValueError(
            f"No numeric descriptor features found (only '{target_column}'). "
            "Full BO requires at least 1 numeric feature column."
        )

    # Ground-truth y (finite only are usable in offline BO)
    y_all = df[target_column].astype(float).values
    finite_mask = np.isfinite(y_all)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size < 3:
        raise ValueError(
            f"Need at least 3 finite values in '{target_column}' to run BO; found {finite_idx.size}."
        )

    # Features with imputation
    X_all = _impute_features(df, feature_cols)

    # Scale features globally (stable across iterations)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    rng = np.random.default_rng(seed)

    # Decide which points are "hidden candidates" vs "initially observed"
    pool_idx = finite_idx.copy()

    if withhold_ids is not None:
        hide_set = set(map(str, withhold_ids))
        cand_idx = np.array([i for i in pool_idx if str(ids_all[i]) in hide_set], dtype=int)
        obs_idx = np.array([i for i in pool_idx if i not in set(cand_idx.tolist())], dtype=int)
        if cand_idx.size == 0:
            raise ValueError("withhold_ids provided, but none matched IDs in the CSV.")
        if obs_idx.size < 2:
            raise ValueError("withhold_ids leaves too few observed points to start BO (need >=2).")
    elif withhold_fraction and withhold_fraction > 0.0:
        frac = float(withhold_fraction)
        frac = max(0.0, min(0.95, frac))
        n_hide = max(1, int(round(frac * pool_idx.size)))
        perm = rng.permutation(pool_idx.size)
        cand_idx = pool_idx[perm[:n_hide]]
        obs_idx = pool_idx[perm[n_hide:]]
        if obs_idx.size < 2:
            raise ValueError("withhold_fraction leaves too few observed points to start BO (need >=2).")
    else:
        # Standard offline BO: start from random initial points, optimise over remaining finite points
        perm = rng.permutation(pool_idx.size)
        n_init_eff = min(int(n_init), pool_idx.size - 1)
        obs_idx = pool_idx[perm[:n_init_eff]]
        cand_idx = pool_idx[perm[n_init_eff:]]

    # Ensure we have enough candidates/observations
    if obs_idx.size < 2:
        raise ValueError(f"Need at least 2 initial observations; got {obs_idx.size}.")
    if cand_idx.size < 1:
        raise ValueError("No candidates left to optimise over (cand_idx empty).")

    # BO trace
    trace_rows: List[Dict[str, object]] = []

    # Kernel / GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)

    def fit_gp(X_obs: np.ndarray, y_obs: np.ndarray) -> GaussianProcessRegressor:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=seed,
            alpha=1e-8,  # numerical jitter
        )
        gp.fit(X_obs, y_obs)
        return gp

    # Current observed set
    obs_set = set(obs_idx.tolist())
    cand_set = set(cand_idx.tolist())

    # Record initial observations
    for i in sorted(obs_set):
        trace_rows.append({
            "step": 0,
            "action": "init_observe",
            "id": str(ids_all[i]),
            "index": int(i),
            "y": float(y_all[i]),
            "acq": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
        })

    # Main BO loop
    budget = int(n_iter)
    for t in range(1, budget + 1):
        obs_list = np.array(sorted(obs_set), dtype=int)
        cand_list = np.array(sorted(cand_set), dtype=int)

        X_obs = X_all_scaled[obs_list]
        y_obs = y_all[obs_list]

        # Fit GP; if it fails, fall back to random candidate
        try:
            gp = fit_gp(X_obs, y_obs)
            mu_c, sigma_c = gp.predict(X_all_scaled[cand_list], return_std=True)
            y_best = float(np.max(y_obs) if maximise else np.min(y_obs))

            acq_name = acquisition.lower().strip()
            if acq_name == "ei":
                acq = _acq_ei(mu_c, sigma_c, y_best=y_best, maximise=maximise, xi=xi)
            elif acq_name == "pi":
                acq = _acq_pi(mu_c, sigma_c, y_best=y_best, maximise=maximise, xi=xi)
            elif acq_name == "ucb":
                acq = _acq_ucb(mu_c, sigma_c, maximise=maximise, kappa=kappa)
            elif acq_name in ("ts", "thompson", "thompson_sampling"):
                acq = _acq_thompson(mu_c, sigma_c, maximise=maximise, rng=rng)
            else:
                raise ValueError(f"Unknown acquisition='{acquisition}'. Use 'ei', 'pi', 'ucb', or 'ts'.")

            j = int(np.argmax(acq))
            chosen = int(cand_list[j])

            chosen_mu = float(mu_c[j])
            chosen_sigma = float(sigma_c[j])
            chosen_acq = float(acq[j])

        except Exception as e:
            # GP/acq failure -> random pick
            j = int(rng.integers(0, cand_list.size))
            chosen = int(cand_list[j])
            chosen_mu = float("nan")
            chosen_sigma = float("nan")
            chosen_acq = float("nan")
            trace_rows.append({
                "step": t,
                "action": "gp_fail_fallback_random",
                "error": str(e),
                "id": str(ids_all[chosen]),
                "index": int(chosen),
                "y": float(y_all[chosen]),
                "acq": chosen_acq,
                "mu": chosen_mu,
                "sigma": chosen_sigma,
            })
            obs_set.add(chosen)
            cand_set.remove(chosen)
            if len(cand_set) == 0:
                break
            continue

        # "Evaluate" chosen candidate (offline reveal of y)
        chosen_y = float(y_all[chosen])

        trace_rows.append({
            "step": t,
            "action": "select_and_observe",
            "id": str(ids_all[chosen]),
            "index": int(chosen),
            "y": chosen_y,
            "acq": chosen_acq,
            "mu": chosen_mu,
            "sigma": chosen_sigma,
        })

        # Update sets
        obs_set.add(chosen)
        cand_set.remove(chosen)

        if len(cand_set) == 0:
            break

    # Final ranking among observed
    obs_list = np.array(sorted(obs_set), dtype=int)
    y_obs = y_all[obs_list]
    order = np.argsort(y_obs)
    if maximise:
        order = order[::-1]

    top_k_eff = min(int(top_k), obs_list.size)
    best_obs = obs_list[order[:top_k_eff]]

    # Save trace if requested
    trace_df = pd.DataFrame(trace_rows)
    if save_trace_csv:
        out_dir = os.path.dirname(save_trace_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        trace_df.to_csv(save_trace_csv, index=False)

    # Build human-readable output
    best_y = float(np.max(y_obs) if maximise else np.min(y_obs))
    lines = []
    lines.append("FULL Bayesian Optimisation (offline) run complete.")
    lines.append(f"Objective: '{target_column}' ({'maximise' if maximise else 'minimise'}).")
    lines.append(f"Features: {len(feature_cols)} numeric descriptor columns.")
    lines.append(f"Finite target pool size: {finite_idx.size}.")
    lines.append(f"Initial observed: {len([r for r in trace_rows if r.get('action') == 'init_observe'])}.")
    lines.append(f"BO steps executed: {len([r for r in trace_rows if r.get('action') == 'select_and_observe'])}.")
    lines.append(f"Best observed value: {best_y:.6g}")
    if save_trace_csv:
        lines.append(f"Trace saved to: {save_trace_csv}")
    lines.append("")
    lines.append(f"Top {top_k_eff} observed IDs after BO:")
    for i in best_obs:
        lines.append(f"- {ids_all[i]}: {target_column} = {y_all[i]:.6g}")

    return "\n".join(lines)


# ===========================
#  JSON entry point
# ===========================

def run_generic_bo_from_config(config_json: str) -> str:
    """
    Full BO entry point.

    Minimal config:
      {
        "csv_path": "C:/.../cof_descriptors_and_props.csv",
        "goal": "Xe/Kr selectivity",
        "maximise": true
      }

    Full BO options:
      - target_column: exact column name (overrides goal resolution)
      - id_column: optional; inferred if omitted
      - top_k: default 10
      - n_init: default 8
      - n_iter: default 30
      - acquisition: "ei" | "ucb" | "pi" | "ts"
      - xi: EI/PI exploration (default 0.01)
      - kappa: UCB exploration (default 2.0)
      - seed: default 0
      - withhold_fraction: simulate unknowns (hide fraction of finite targets)
      - withhold_ids: explicit list of IDs to hide (overrides fraction)
      - save_trace_csv: where to write a per-step trace (prevents overwrite if you name it uniquely)
    """
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as e:
        return f"Failed to parse JSON config: {e}"

    csv_path = cfg.get("csv_path") or cfg.get("descriptor_csv")
    if not csv_path:
        return "Config must include 'csv_path' pointing to the CSV file."

    goal = cfg.get("goal")
    target_column = cfg.get("target_column")
    id_column = cfg.get("id_column")

    maximise = bool(cfg.get("maximise", True))
    top_k = int(cfg.get("top_k", 10))

    n_init = int(cfg.get("n_init", 8))
    n_iter = int(cfg.get("n_iter", 30))
    acquisition = str(cfg.get("acquisition", "ei"))
    xi = float(cfg.get("xi", 0.01))
    kappa = float(cfg.get("kappa", 2.0))
    seed = int(cfg.get("seed", 0))

    withhold_fraction = float(cfg.get("withhold_fraction", 0.0))
    withhold_ids = cfg.get("withhold_ids", None)
    save_trace_csv = cfg.get("save_trace_csv", None)

    try:
        df = _load_csv(csv_path)
        resolved_target = _resolve_objective_column(df, goal=goal, explicit=target_column)

        header = [
            f"Loaded CSV: {csv_path}",
            f"Rows × cols: {df.shape[0]} × {df.shape[1]}",
            f"Resolved objective column: '{resolved_target}' (goal='{goal}', target_column='{target_column}')",
            "",
        ]

        out = _run_full_offline_bo(
            df=df,
            id_column=id_column,
            target_column=resolved_target,
            maximise=maximise,
            top_k=top_k,
            n_init=n_init,
            n_iter=n_iter,
            acquisition=acquisition,
            xi=xi,
            kappa=kappa,
            seed=seed,
            withhold_fraction=withhold_fraction,
            withhold_ids=withhold_ids,
            save_trace_csv=save_trace_csv,
        )

        return "\n".join(header) + out

    except Exception as e:
        return f"Error during FULL BO: {e}"


# ===========================
#  LangChain / ChemCrow tool
# ===========================

class GenericBayesOpt1D(BaseTool):
    """
    FULL (offline) Bayesian Optimisation over a single CSV table.

    This is BO in the proper sense: it iteratively selects candidates via an
    acquisition function (EI/UCB/PI/TS), reveals their true objective value
    from the CSV (offline evaluation), retrains the GP, and repeats.

    Use cases:
      - Offline benchmarking: does BO find the best COFs faster than random search?
      - Practical ranking: produce top-k IDs after a fixed evaluation budget.

    Important:
      - Full offline BO requires the target column to be present (finite) for a
        meaningful pool of rows. You can simulate unknowns via withhold_fraction/withhold_ids.
    """

    name = "GenericBayesOpt1D"
    description = (
        "FULL (offline) Bayesian Optimisation over a single CSV. "
        "Input MUST be JSON with 'csv_path' and either 'target_column' or a free-text 'goal'. "
        "Optional: 'id_column', 'maximise'(default true), 'top_k'(default 10), "
        "'n_init'(default 8), 'n_iter'(default 30), acquisition='ei'|'ucb'|'pi'|'ts', "
        "'xi'(EI/PI), 'kappa'(UCB), 'seed', "
        "'withhold_fraction' or 'withhold_ids' to simulate unknown targets, "
        "and 'save_trace_csv' to persist a per-step trace."
    )

    llm: Optional[BaseLanguageModel] = None

    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__()
        self.llm = llm

    def _run(self, query: str) -> str:
        return run_generic_bo_from_config(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async.")

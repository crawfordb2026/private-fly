#!/usr/bin/env python3
"""
Pipeline Step 2.5: HMM Health State Detection

R/hmm.r-matched logic:
- Bin to 5-minute windows and compute MT_sum and Pn_var per bin.
- Prelabel per-bin states (Sleep / Warmup / Low_Data / Dead).
- Compute per-fly rolling baseline z-scores on Active bins only.
- Fit an unsupervised 3-state Gaussian HMM per genotype on `HMM_candidate` bins
  (grouped into sufficiently-long candidate bouts).
- Map decoded HMM states to (Healthy, Declining, Critical).
- Output `hmm_states` using integer encoding 0..3: Healthy, Declining, Critical, Dead.
"""

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from hmmlearn import hmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_CONFIG, DATABASE_URL

# ==================== CONSTANTS ====================
EXPERIMENT_ID = None  # None = auto-detect latest experiment

BIN_MINUTES = 5  # aggregate raw minute counts into bins

# Output encoding into `hmm_states`:
#   0=Healthy, 1=Declining, 2=Critical, 3=Dead
N_STATES = 4
HMM_N_STATES = 3
STATE_NAMES = ['Healthy', 'Declining', 'Critical', 'Dead']
STATE_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
N_PLOT = 48  # default example flies (stratified by genotype); override with --plots

# R/hmm.r constants (all defined on 5-minute bins)
MIN_BOUT_BINS = 6     # 30 minutes: short Active bouts relabeled as Sleep
DEATH_BINS = 144      # 12 hours: first run of 144 bins with MT_sum==0 and Pn_var==0
ROLL_BINS = 288       # 24 hours: rolling baseline over trailing ACTIVE bins only
WARMUP_ACTIVE = 144  # warmup: first 12 hours worth of ACTIVE bins excluded from HMM
SD_FLOOR_MULT = 0.01  # rolling SD below 1% of fly's own active SD => Low_Data

# Kept for backward compatibility with older unused code paths.
DEATH_THRESHOLD_HOURS = 12
DEATH_FILTER_HOURS = 12

VALID_STATE_LABELS = (
    "Healthy",
    "Declining",
    "Critical",
    "Dead",
    "Sleep",
    "Warmup",
    "Low_Data",
)


# ==================== DATA LOADING ====================
def get_experiment_id():
    """Use EXPERIMENT_ID constant, or fall back to latest experiment."""
    if EXPERIMENT_ID is not None:
        return EXPERIMENT_ID
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1")
    eid = cur.fetchone()[0]
    conn.close()
    return eid


def load_all_readings(experiment_id):
    """Load MT and Pn readings. Returns dict: fly_id -> (mt_array, pn_array)."""
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT fly_id, reading_type, value FROM readings "
        "WHERE experiment_id = %(eid)s AND reading_type IN ('MT', 'Pn') "
        "ORDER BY fly_id, datetime",
        engine, params={'eid': experiment_id}
    )
    engine.dispose()
    result = {}
    for fid, grp in df.groupby('fly_id'):
        mt = grp[grp['reading_type'] == 'MT']['value'].values
        pn = grp[grp['reading_type'] == 'Pn']['value'].values
        if len(mt) > 0 and len(pn) > 0:
            n = min(len(mt), len(pn))
            result[fid] = (mt[:n], pn[:n])
    return result


def load_genotypes(experiment_id):
    """Returns dict: fly_id -> genotype."""
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT fly_id, genotype FROM flies WHERE experiment_id = %(eid)s",
        engine, params={'eid': experiment_id}
    )
    engine.dispose()
    return dict(zip(df['fly_id'], df['genotype']))


# ==================== PREPROCESSING ====================
def bin_features(mt_raw, pn_raw, bin_min=BIN_MINUTES):
    """Bin into 2 features: MT sum and Pn variance per bin. Returns (n_bins, 2) array."""
    n = min(len(mt_raw), len(pn_raw)) // bin_min
    mt = mt_raw[:n * bin_min].reshape(n, bin_min).sum(axis=1).astype(float)
    pn = pn_raw[:n * bin_min].reshape(n, bin_min).astype(float)
    pn_var = np.var(pn, axis=1)
    return np.column_stack([mt, pn_var])


def find_death_boundary(raw, threshold_hours=DEATH_THRESHOLD_HOURS):
    """First minute index where threshold_hours of continuous zero activity begins.
    Returns None if the fly never died."""
    th = threshold_hours * 60
    if len(raw) < th:
        return None
    is_zero = (raw == 0).astype(np.int32)
    cs = np.cumsum(np.concatenate([[0], is_zero]))
    rolling_zeros = cs[th:] - cs[:-th]
    hits = np.where(rolling_zeros == th)[0]
    return int(hits[0]) if len(hits) > 0 else None


# ==================== Z-SCORE NORMALIZATION ====================
def find_death_mask(mt_binned, threshold_hours=DEATH_FILTER_HOURS):
    """Boolean mask: True for bins that are part of a >= threshold_hours zero stretch."""
    threshold_bins = threshold_hours * 60 // BIN_MINUTES
    is_zero = (mt_binned == 0).astype(np.int32)
    mask = np.zeros(len(mt_binned), dtype=bool)
    streak_start = None
    for i in range(len(is_zero)):
        if is_zero[i]:
            if streak_start is None:
                streak_start = i
        else:
            if streak_start is not None and (i - streak_start) >= threshold_bins:
                mask[streak_start:i] = True
            streak_start = None
    if streak_start is not None and (len(is_zero) - streak_start) >= threshold_bins:
        mask[streak_start:] = True
    return mask


def compute_zscore_stats(features, genotypes, fly_ids):
    """Compute per-genotype mean/std from alive periods only."""
    geno_data = {}
    for fid in fly_ids:
        g = genotypes.get(fid)
        if g is None:
            continue
        feat = features[fid]
        alive_mask = ~find_death_mask(feat[:, 0])
        if alive_mask.any():
            geno_data.setdefault(g, []).append(feat[alive_mask])

    stats = {}
    for g, arrays in geno_data.items():
        stacked = np.vstack(arrays)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        std[std < 1e-6] = 1.0  # avoid divide-by-zero
        stats[g] = (mean, std)
        print(f'  {g}: mean=[{mean[0]:.1f}, {mean[1]:.2f}]  std=[{std[0]:.1f}, {std[1]:.2f}]  '
              f'({len(arrays)} flies, {stacked.shape[0]} alive bins)')
    return stats


def apply_zscore(features, genotypes, stats):
    """Z-score all flies using their genotype's stats. Returns new dict."""
    normed = {}
    for fid, feat in features.items():
        g = genotypes.get(fid)
        if g is None or g not in stats:
            normed[fid] = feat  # no genotype info, pass through
            continue
        mean, std = stats[g]
        normed[fid] = (feat - mean) / std
    return normed


# ==================== R-STYLE PREPROCESSING ====================
def load_binned_mt_pn_and_metadata(experiment_id: int) -> pd.DataFrame:
    """
    Load minute-level MT/Pn from DB, pivot to wide (MT,Pn), keep only complete minutes,
    then bin to 5-minute windows. Only bins with `n_obs == BIN_MINUTES` are kept
    (matching R/hmm.r's `filter(n_obs == 5)`).

    Returns a dataframe with one row per fly/bin:
      fly_id, bin_start, genotype, sex, treatment, MT_sum, Pn_var
    """
    engine = create_engine(DATABASE_URL)

    df = pd.read_sql(
        """
        SELECT
            r.fly_id,
            r.datetime,
            r.reading_type,
            r.value,
            f.genotype,
            f.sex,
            f.treatment
        FROM readings r
        JOIN flies f
          ON f.fly_id = r.fly_id AND f.experiment_id = r.experiment_id
        WHERE r.experiment_id = %(eid)s
          AND r.reading_type IN ('MT', 'Pn')
        ORDER BY r.fly_id, r.datetime
        """,
        engine,
        params={"eid": experiment_id},
    )
    engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])

    wide = (
        df.pivot_table(
            index=["fly_id", "datetime", "genotype", "sex", "treatment"],
            columns="reading_type",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Keep only minutes where both channels are present (R: filter(!is.na(MT), !is.na(Pn)))
    wide = wide.dropna(subset=["MT", "Pn"])

    # Bin to 5-minute starts
    wide["bin_start"] = wide["datetime"].dt.floor(f"{BIN_MINUTES}min")

    def pn_var_ddof1(x: pd.Series) -> float:
        # numpy var uses ddof; R's var() uses sample variance (ddof=1) by default.
        return float(np.var(x.to_numpy(dtype=float), ddof=1))

    binned = (
        wide.groupby(["fly_id", "bin_start", "genotype", "sex", "treatment"], sort=False)
        .agg(
            n_obs=("datetime", "size"),
            MT_sum=("MT", "sum"),
            Pn_var=("Pn", pn_var_ddof1),
        )
        .reset_index()
    )

    # R: filter(n_obs == 5)
    binned = binned[binned["n_obs"] == BIN_MINUTES].copy()
    binned["MT_sum"] = binned["MT_sum"].astype(float)
    binned["Pn_var"] = binned["Pn_var"].astype(float).fillna(0.0)

    binned = binned.sort_values(["fly_id", "bin_start"]).reset_index(drop=True)
    return binned[["fly_id", "bin_start", "genotype", "sex", "treatment", "MT_sum", "Pn_var"]]


def _find_first_run_start(is_true: np.ndarray, run_len: int) -> int | None:
    """Return the first index i such that is_true[i:i+run_len] are all True."""
    if run_len <= 0 or is_true.size < run_len:
        return None
    for i in range(0, is_true.size - run_len + 1):
        if np.all(is_true[i : i + run_len]):
            return i
    return None


def _relabel_short_active_bouts(raw_state: np.ndarray, min_bout_bins: int) -> np.ndarray:
    """
    R: relabel active bouts shorter than `min_bout_bins` as Sleep.
    """
    raw_state = raw_state.astype(object).copy()
    if raw_state.size == 0:
        return raw_state

    # Run-length id per consecutive identical value
    changes = raw_state != np.concatenate(([raw_state[0]], raw_state[:-1]))
    rle_id = np.cumsum(changes)
    _, counts = np.unique(rle_id, return_counts=True)
    bout_len_by_rle = dict(zip(np.unique(rle_id), counts))
    bout_lens = np.array([bout_len_by_rle[i] for i in rle_id], dtype=int)

    short_active = (raw_state == "Active") & (bout_lens < min_bout_bins)
    raw_state[short_active] = "Sleep"
    return raw_state


def preprocess_r_states(binned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R/hmm.r style per-bin labels and rolling z-scores:
      - raw_state: Dead / Sleep / Active (includes short-bout relabeling)
      - MT_roll_z, Pn_roll_z
      - hmm_state: Dead / Sleep / Warmup / Low_Data / HMM_candidate
    """
    if binned_df.empty:
        binned_df = binned_df.copy()
        binned_df["raw_state"] = []
        binned_df["hmm_state"] = []
        binned_df["MT_roll_z"] = np.nan
        binned_df["Pn_roll_z"] = np.nan
        return binned_df

    out = []
    for fly_id, g in binned_df.groupby("fly_id", sort=False):
        g = g.sort_values("bin_start").copy()
        mt_sum = g["MT_sum"].to_numpy(dtype=float)
        pn_var = g["Pn_var"].to_numpy(dtype=float)
        bin_start = g["bin_start"].to_numpy()

        # Death = first onset of DEATH_BINS consecutive bins where MT_sum==0 and Pn_var==0
        is_dead_bin = (mt_sum == 0.0) & (pn_var == 0.0)
        tod_idx = _find_first_run_start(is_dead_bin, DEATH_BINS)
        time_of_death = bin_start[tod_idx] if tod_idx is not None else None

        is_post_death = False
        if time_of_death is not None:
            is_post_death = bin_start >= time_of_death

        raw_state = np.where(is_post_death, "Dead", np.where(mt_sum == 0.0, "Sleep", "Active"))
        raw_state = _relabel_short_active_bouts(raw_state, MIN_BOUT_BINS)

        active_cumcount = np.cumsum(raw_state == "Active")

        # Fly-level SD over ACTIVE bins for SD flooring
        active_mask = raw_state == "Active"
        fly_mt_sd = float(np.std(mt_sum[active_mask], ddof=1)) if np.sum(active_mask) > 1 else np.nan
        fly_pn_sd = float(np.std(pn_var[active_mask], ddof=1)) if np.sum(active_mask) > 1 else np.nan
        mt_sd_floor = SD_FLOOR_MULT * fly_mt_sd
        pn_sd_floor = SD_FLOOR_MULT * fly_pn_sd

        if not np.isfinite(mt_sd_floor) or mt_sd_floor <= 0:
            mt_sd_floor = 1e-6
        if not np.isfinite(pn_sd_floor) or pn_sd_floor <= 0:
            pn_sd_floor = 1e-6

        # Rolling baseline stats:
        # R/hmm.r uses `rollapply(..., width=ROLL_BINS, align="right")` over BIN INDICES
        # and then computes mean/sd on only the Active bins within that time window.
        mt_roll_sd = np.full(raw_state.size, np.nan, dtype=float)
        pn_roll_sd = np.full(raw_state.size, np.nan, dtype=float)
        mt_roll_z = np.full(raw_state.size, np.nan, dtype=float)
        pn_roll_z = np.full(raw_state.size, np.nan, dtype=float)

        active_idx = deque()
        active_mt = deque()
        active_pn = deque()
        mt_sum_w = 0.0
        mt_sum2_w = 0.0
        pn_sum_w = 0.0
        pn_sum2_w = 0.0

        for i in range(raw_state.size):
            if raw_state[i] == "Active":
                v_mt = float(mt_sum[i])
                v_pn = float(pn_var[i])
                active_idx.append(i)
                active_mt.append(v_mt)
                active_pn.append(v_pn)
                mt_sum_w += v_mt
                mt_sum2_w += v_mt * v_mt
                pn_sum_w += v_pn
                pn_sum2_w += v_pn * v_pn

            # Keep Active points within the trailing time window of size ROLL_BINS.
            cutoff = i - ROLL_BINS + 1
            while active_idx and active_idx[0] < cutoff:
                old_mt = active_mt.popleft()
                old_pn = active_pn.popleft()
                active_idx.popleft()
                mt_sum_w -= old_mt
                mt_sum2_w -= old_mt * old_mt
                pn_sum_w -= old_pn
                pn_sum2_w -= old_pn * old_pn

            if raw_state[i] != "Active":
                continue

            n_active_in_window = len(active_idx)
            mt_mean = mt_sum_w / n_active_in_window
            pn_mean = pn_sum_w / n_active_in_window

            if n_active_in_window >= 2:
                mt_var = (mt_sum2_w - (mt_sum_w * mt_sum_w) / n_active_in_window) / (n_active_in_window - 1)
                pn_window_var = (pn_sum2_w - (pn_sum_w * pn_sum_w) / n_active_in_window) / (n_active_in_window - 1)
                mt_sd = float(np.sqrt(max(mt_var, 0.0)))
                pn_sd = float(np.sqrt(max(pn_window_var, 0.0)))
            else:
                mt_sd = np.nan
                pn_sd = np.nan

            mt_roll_sd[i] = mt_sd
            pn_roll_sd[i] = pn_sd

            # R: denom uses pmax(MT_roll_sd, mt_sd_floor, na.rm=TRUE)
            mt_denom = mt_sd if np.isfinite(mt_sd) and mt_sd > 0 else mt_sd_floor
            pn_denom = pn_sd if np.isfinite(pn_sd) and pn_sd > 0 else pn_sd_floor

            if not np.isfinite(mt_denom) or mt_denom <= 0:
                mt_denom = 1e-6
            if not np.isfinite(pn_denom) or pn_denom <= 0:
                pn_denom = 1e-6

            mt_roll_z[i] = (mt_sum[i] - mt_mean) / mt_denom
            pn_roll_z[i] = (pn_var[i] - pn_mean) / pn_denom

        # Final hmm_state label per bin
        hmm_state = np.full(raw_state.size, "Sleep", dtype=object)
        hmm_state[raw_state == "Dead"] = "Dead"
        hmm_state[raw_state == "Sleep"] = "Sleep"

        warmup_mask = (raw_state == "Active") & (active_cumcount < WARMUP_ACTIVE)
        hmm_state[warmup_mask] = "Warmup"

        low_data_mask = (raw_state == "Active") & ~warmup_mask & (
            (np.isfinite(mt_roll_sd) & (mt_roll_sd < mt_sd_floor))
            | (np.isfinite(pn_roll_sd) & (pn_roll_sd < pn_sd_floor))
        )
        hmm_state[low_data_mask] = "Low_Data"

        hmm_state[(raw_state == "Active") & ~warmup_mask & ~low_data_mask] = "HMM_candidate"

        g["raw_state"] = raw_state
        g["active_cumcount"] = active_cumcount
        g["MT_roll_sd"] = mt_roll_sd
        g["Pn_roll_sd"] = pn_roll_sd
        g["MT_roll_z"] = mt_roll_z
        g["Pn_roll_z"] = pn_roll_z
        g["hmm_state"] = hmm_state
        out.append(g)

    return pd.concat(out, axis=0, ignore_index=True)


# ==================== R-STYLE GENOTYPE HMM FITTING ====================
def _get_start_vals(values: np.ndarray, spread_factor: float) -> np.ndarray:
    """
    R/hmm.r:
      lo   <- max(0.05, 0.15 - 0.10 * (spread_factor - 1))
      hi   <- min(0.95, 0.85 + 0.10 * (spread_factor - 1))
      qs <- quantile(values, probs=c(lo,0.50,hi), na.rm=TRUE)
    """
    lo = max(0.05, 0.15 - 0.10 * (spread_factor - 1.0))
    hi = min(0.95, 0.85 + 0.10 * (spread_factor - 1.0))
    probs = [lo, 0.50, hi]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return np.quantile(values, probs)


def fit_decode_genotype_hmm(candidate_df: pd.DataFrame, max_tries: int = 4):
    """
    Fit a 3-state Gaussian HMM per R logic:
    - candidate bouts: break when time_gap > 5 minutes
    - keep only bouts with length >= MIN_BOUT_BINS
    - retry EM with different initial emission means (spread_factor)
    - map decoded state indices to (Healthy,Declining,Critical) by emission means order

    Returns:
      (index_array, state_int_codes_array) or None on failure/insufficient data.
    """
    if candidate_df.empty:
        return None

    cand = candidate_df.copy()
    if "fly_id" not in cand.columns or "bin_start" not in cand.columns:
        raise RuntimeError("HMM candidate dataframe must include columns fly_id and bin_start.")

    cand = cand.dropna(subset=["MT_roll_z", "Pn_roll_z"]).sort_values(["fly_id", "bin_start"]).copy()

    if cand.empty:
        return None

    # Candidate bout structure within each fly.
    # Use vectorized groupby-diff/cumsum instead of groupby-apply so `fly_id`
    # remains a normal column across pandas versions.
    gaps_mins = (
        cand.groupby("fly_id", sort=False)["bin_start"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
    )
    gap_break = gaps_mins > 5.0
    cand["bout_id"] = gap_break.groupby(cand["fly_id"], sort=False).cumsum().astype(int)

    bout_lengths = (
        cand.groupby(["fly_id", "bout_id"], sort=False)
        .size()
        .reset_index(name="n")
    )
    bout_lengths = bout_lengths[bout_lengths["n"] >= MIN_BOUT_BINS].copy()
    if bout_lengths.empty:
        return None

    cand_fit = cand.merge(bout_lengths[["fly_id", "bout_id"]], on=["fly_id", "bout_id"], how="inner")
    cand_fit = cand_fit.sort_values(["fly_id", "bout_id", "bin_start"]).copy()

    if len(cand_fit) < 200:
        return None

    # ntimes_vec is the per-bout sequence lengths in order
    bout_order = (
        cand_fit.groupby(["fly_id", "bout_id"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["fly_id", "bout_id"])
    )
    lengths = bout_order["n"].astype(int).tolist()
    if not lengths:
        return None

    X = cand_fit[["MT_roll_z", "Pn_roll_z"]].to_numpy(dtype=float)

    mt_all = cand_fit["MT_roll_z"].to_numpy(dtype=float)
    pn_all = cand_fit["Pn_roll_z"].to_numpy(dtype=float)

    # EM retry loop (R: fit_with_retry)
    for attempt in range(1, max_tries + 1):
        spread = 1.0 + (attempt - 1) * 0.5
        mt_q = _get_start_vals(mt_all, spread)
        pn_q = _get_start_vals(pn_all, spread)

        mt_var = max(float(np.var(mt_all)), 1e-3)
        pn_var = max(float(np.var(pn_all)), 1e-3)

        model = hmm.GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="diag",
            n_iter=500,
            tol=1e-8,
            random_state=42,
            params="stmc",
            init_params="",
        )
        # depmixS4 doesn't constrain transitions here; keep them free.
        model.startprob_ = np.full(HMM_N_STATES, 1.0 / HMM_N_STATES, dtype=float)
        model.transmat_ = np.full((HMM_N_STATES, HMM_N_STATES), 1.0 / HMM_N_STATES, dtype=float)
        model.means_ = np.column_stack([mt_q, pn_q])
        model.covars_ = np.tile([mt_var, pn_var], (HMM_N_STATES, 1))

        try:
            model.fit(X, lengths)
        except Exception:
            continue

        converged = True
        if hasattr(model, "monitor_") and hasattr(model.monitor_, "converged"):
            converged = bool(model.monitor_.converged)
        if not converged:
            continue

        # Decode each bout separately to respect sequence boundaries
        decoded_states = []
        start = 0
        for L in lengths:
            seg = X[start : start + L]
            decoded_states.append(model.predict(seg))
            start += L
        decoded_states = np.concatenate(decoded_states)

        # Map HMM component indices to R's labels by emission means ordering
        mt_means = np.array(
            [np.mean(mt_all[decoded_states == k]) for k in range(HMM_N_STATES)], dtype=float
        )
        pn_means = np.array(
            [np.mean(pn_all[decoded_states == k]) for k in range(HMM_N_STATES)], dtype=float
        )
        mt_means = np.where(np.isfinite(mt_means), mt_means, -np.inf)
        pn_means = np.where(np.isfinite(pn_means), pn_means, -np.inf)

        order = sorted(range(HMM_N_STATES), key=lambda k: (-mt_means[k], -pn_means[k]))
        # R mapping: best -> Healthy, next -> Declining, last -> Critical
        code_map = {order[0]: 0, order[1]: 1, order[2]: 2}
        state_codes = np.array([code_map[int(s)] for s in decoded_states], dtype=int)

        return cand_fit.index.to_numpy(), state_codes

    return None


# ==================== HMM ====================
def train_hmm(features_list):
    """Unsupervised left-to-right 4-state GaussianHMM on 2 features (MT sum, Pn var).

    Emission means initialized from data quantiles. Transition matrix fixed.
    Baum-Welch learns only the emission parameters.
    """
    all_data = np.vstack(features_list)  # (total_bins, 2)
    mt_all, pn_all = all_data[:, 0], all_data[:, 1]

    # quantile-based init for each feature
    mt_q = np.percentile(mt_all, [75, 50, 25, 5])
    pn_q = np.percentile(pn_all, [75, 50, 25, 5])
    mt_var = max(float(np.var(mt_all)), 0.01)
    pn_var = max(float(np.var(pn_all)), 0.01)

    model = hmm.GaussianHMM(
        n_components=N_STATES, covariance_type='diag',
        n_iter=100, params='mc', init_params='', random_state=42, #mct
    )
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
    model.transmat_ = np.array([
        [0.95, 0.05, 0.00, 0.00],   # Healthy -> Healthy | Declining
        [0.00, 0.85, 0.15, 0.00],   # Declining -> Declining | Critical
        [0.00, 0.00, 0.90, 0.10],   # Critical -> Critical | Dead
        [0.00, 0.00, 0.00, 1.00],   # Dead (absorbing)
    ])
    model.means_ = np.array([
        [mt_q[0], pn_q[0]],   # Healthy: high MT, high Pn variance
        [mt_q[1], pn_q[1]],   # Declining
        [mt_q[2], pn_q[2]],   # Critical
        [mt_q[3], pn_q[3]],   # Dead: near-zero both
    ])
    model.covars_ = np.array([
        [mt_var, pn_var],
        [mt_var, pn_var],
        [mt_var, pn_var],
        [0.01,   0.01],
    ])

    print(f'  Init means:  MT={[f"{m:.1f}" for m in mt_q]}  Pn={[f"{m:.1f}" for m in pn_q]}')

    X = np.vstack(features_list)
    lengths = [len(f) for f in features_list]
    model.fit(X, lengths)

    print(f'  Trained means (MT, Pn_var):')
    for i, name in enumerate(STATE_NAMES):
        print(f'    {name:>10}: MT={model.means_[i,0]:.1f}  Pn={model.means_[i,1]:.2f}')
    return model


def predict_states(model, features):
    """Viterbi decode with post-hoc Dead-absorbing enforcement."""
    states = model.predict(features)
    first_dead = np.where(states == 3)[0]
    if len(first_dead) > 0:
        states[first_dead[0]:] = 3
    return states


# ==================== EVALUATION ====================
def evaluate(all_states, death_bins, fly_ids):
    """Return list of advance-warning hours for each dead fly."""
    advance = []
    for fid in fly_ids:
        db = death_bins.get(fid)
        if db is None:
            continue
        states = all_states[fid]
        warn = np.where((states == 2) | (states == 3))[0]
        if len(warn) > 0:
            advance.append((db - warn[0]) * BIN_MINUTES / 60)
    return advance


# ==================== VISUALIZATION ====================
def plot_fly(features, states, death_bin, fly_id, save_dir='hmm_plots'):
    """MT activity, Pn variance, and state assignments for one fly."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    hours = np.arange(len(features)) * BIN_MINUTES / 60

    for ax in (ax1, ax2):
        if death_bin is not None:
            ax.axvline(death_bin * BIN_MINUTES / 60, color='red', ls='--', alpha=0.5)

    ax1.plot(hours, features[:, 0], 'k-', alpha=0.7, lw=0.5)
    ax1.set_ylabel('MT activity')
    ax1.set_title(f'Fly {fly_id}')

    ax2.plot(hours, features[:, 1], 'k-', alpha=0.7, lw=0.5)
    ax2.set_ylabel('Pn variance')

    for i in range(len(states)):
        x0 = hours[i]
        x1 = hours[i + 1] if i + 1 < len(hours) else x0 + BIN_MINUTES / 60
        ax3.axvspan(x0, x1, color=STATE_COLORS[states[i]], alpha=0.7)
    ax3.set_yticks(range(N_STATES))
    ax3.set_yticklabels(STATE_NAMES)
    ax3.set_ylabel('State')
    ax3.set_xlabel('Hours')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fly_{fly_id.replace("/", "-")}_hmm.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==================== DATABASE ====================
def save_results(experiment_id, all_states, all_state_labels):
    """Replace hmm_states rows for this experiment (table must match schema.sql)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("DELETE FROM hmm_states WHERE experiment_id = %s", [experiment_id])

    insert_sql = (
        "INSERT INTO hmm_states "
        "(experiment_id, fly_id, bin_index, bin_minutes, state, state_label) VALUES %s"
    )
    rows = []
    for fly_id, states in all_states.items():
        labels = all_state_labels[fly_id]
        for i, s in enumerate(states):
            rows.append((experiment_id, fly_id, i, BIN_MINUTES, int(s), labels[i]))
        if len(rows) >= 10_000:
            execute_values(cur, insert_sql, rows)
            rows = []
    if rows:
        execute_values(cur, insert_sql, rows)

    conn.commit()
    conn.close()


# ==================== MAIN ====================
if __name__ == '__main__':
    _parser = argparse.ArgumentParser(
        description="HMM health states (R/hmm.r-aligned). Database from config/env."
    )
    _parser.add_argument(
        "--plots",
        type=int,
        default=None,
        metavar="N",
        help=f"number of example PNGs (default {N_PLOT}); random sample of flies",
    )
    _parser.add_argument(
        "--plot-seed",
        type=int,
        default=42,
        help="RNG seed for stratified plot selection (default 42)",
    )
    _args = _parser.parse_args()
    _n_plot = _args.plots if _args.plots is not None else N_PLOT
    _plot_seed = _args.plot_seed

    eid = get_experiment_id()
    print(f'Experiment: {eid}')

    # --- load + bin (R-style) ---
    print('Loading MT + Pn and binning to 5-min windows...')
    binned_df = load_binned_mt_pn_and_metadata(eid)
    if binned_df.empty:
        raise RuntimeError("No complete 5-min bins found in DB for this experiment.")

    n_flies = int(binned_df["fly_id"].nunique())
    print(f'  Loaded {len(binned_df):,} bins across {n_flies} flies')

    # --- compute R-style prelabels + rolling z-scores ---
    print('Precomputing R/hmm.r states (Sleep/Warmup/Low_Data/Dead) and rolling z-scores...')
    df_states = preprocess_r_states(binned_df)

    # Output state encoding for `hmm_states`
    # NOTE: DB schema only stores 0..3. We use:
    #   - Dead => 3
    #   - HMM_candidate decoded bins => {Healthy:0, Declining:1, Critical:2}
    #   - Sleep/Warmup/Low_Data => 0 (placeholder), but preserve label text
    #   - Any unresolved HMM_candidate (e.g., failed/insufficient decode) => Healthy
    df_states["state_label"] = df_states["hmm_state"].astype(str)
    df_states["state_int"] = 0
    df_states.loc[df_states["hmm_state"] == "Dead", "state_int"] = 3

    # --- fit HMM per genotype on HMM_candidate bouts ---
    genotypes = sorted(df_states["genotype"].dropna().unique().tolist())
    failed_genotypes = []

    print(f'Fitting genotype-specific HMMs for {len(genotypes)} genotypes...')
    for i, geno in enumerate(genotypes, start=1):
        cand = df_states[
            (df_states["genotype"] == geno) & (df_states["hmm_state"] == "HMM_candidate")
        ].copy()
        cand = cand.dropna(subset=["MT_roll_z", "Pn_roll_z"])

        if cand.empty or len(cand) < 200:
            failed_genotypes.append(geno)
            continue

        decoded = fit_decode_genotype_hmm(cand)
        if decoded is None:
            failed_genotypes.append(geno)
            continue

        idx, codes = decoded
        df_states.loc[idx, "state_int"] = codes
        df_states.loc[idx, "state_label"] = pd.Series(codes, index=idx).map({
            0: "Healthy",
            1: "Declining",
            2: "Critical",
        }).values

        if i % 10 == 0:
            print(f'  Progress: {i}/{len(genotypes)} genotypes')

    if failed_genotypes:
        print(f'Genotypes skipped/failed: {", ".join(failed_genotypes[:20])}'
              + ("" if len(failed_genotypes) <= 20 else f' ... (+{len(failed_genotypes)-20} more)'))

    # Keep labels DB-valid even when some candidate bins were not decoded.
    unresolved_mask = df_states["state_label"] == "HMM_candidate"
    unresolved_n = int(unresolved_mask.sum())
    if unresolved_n > 0:
        df_states.loc[unresolved_mask, "state_label"] = "Healthy"
        df_states.loc[unresolved_mask, "state_int"] = 0
        print(f'  Unresolved candidate bins mapped to Healthy: {unresolved_n:,}')

    # --- assemble per-fly sequences ---
    all_states = {}
    all_state_labels = {}
    features_for_plot = {}
    for fid, g in df_states.groupby("fly_id", sort=False):
        g = g.sort_values("bin_start")
        seq = g["state_int"].astype(int).to_numpy()
        labels = g["state_label"].astype(str).to_numpy()
        bad = set(labels) - set(VALID_STATE_LABELS)
        if bad:
            raise RuntimeError(f"Unexpected state labels for fly {fid}: {sorted(bad)}")
        all_states[fid] = seq
        all_state_labels[fid] = labels
        features_for_plot[fid] = g[["MT_sum", "Pn_var"]].to_numpy(dtype=float)

    print(f'  Decoded state sequences for {len(all_states)} flies')

    # --- plot examples (random subset; use --plot-seed to reproduce) ---
    death_first_index = {}
    for fid, seq in all_states.items():
        dead_idx = np.where(seq == 3)[0]
        death_first_index[fid] = int(dead_idx[0]) if dead_idx.size > 0 else None

    plot_ids = list(all_states.keys())
    np.random.default_rng(_plot_seed).shuffle(plot_ids)
    plot_ids = plot_ids[:_n_plot]
    print(f"\nPlotting {len(plot_ids)} example flies (seed={_plot_seed})...")
    for fid in plot_ids:
        plot_fly(
            features_for_plot[fid],
            all_states[fid],
            death_first_index.get(fid),
            fid,
        )
        print(f'  Plotted {fid}')

    # --- save to DB ---
    print('\nSaving states to database...')
    save_results(eid, all_states, all_state_labels)

    # --- summary ---
    all_s = np.concatenate(list(all_states.values()))
    print(f'\n{"=" * 50}')
    print(f'DONE - {len(all_states)} flies processed')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:>10}: {np.mean(all_s == i):6.1%} of time bins')
    print(f'  Output: hmm_states table (state + state_label), hmm_plots/')

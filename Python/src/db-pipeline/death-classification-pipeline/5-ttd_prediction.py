#!/usr/bin/env python3
"""
Pipeline Step 5: Time-to-Death (TTD) Prediction

Trains a Random Forest regressor to predict hours remaining until death
from 6-hour sleep feature windows anchored backward from each fly's death time.

TTD convention: positive hours remaining.
  TTD=6  → fly dies within the next 6 hours
  TTD=12 → fly has 12 hours remaining
  TTD increases as windows go further back from death.

Windows are non-overlapping, 6 hours wide, anchored backward from death.
Windows with < MIN_DATA_HOURS of actual data coverage are discarded.

Features per window (11 total fed to the model):
  9 sleep features  +  ZT_sin  +  ZT_cos

Cross-validation: group k-fold (k=5), groups=fly_id, stratified by genotype.
Genotype is NOT a model feature — only used for fold stratification.
TTD is the sole regression target and is never an input feature.

Outputs (analysis/analysis_results/):
  windowed_features.csv     full feature matrix: fly_id, TTD, ZT_sin, ZT_cos, 9 features
  model_performance.txt     MAE, RMSE, R² per fold and averaged
  feature_importances.csv   importances ranked highest-to-lowest, mean ± std across folds
  predicted_vs_actual.png   scatter plot colored by fold with diagonal reference line

Usage:
  python 5-ttd_prediction.py
  python 5-ttd_prediction.py --experiment-id 1 --n-folds 5
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
#   CONFIGURATION — change values here, nowhere else
# ============================================================

# Death = start of the first run of this many consecutive hours with zero MT activity.
# TTD is measured from this point backward.
DEATH_IMMOBILITY_HOURS = 24

# Minimum hours of actual experiment coverage a window must have to be kept.
# A window that straddles the start of monitoring is discarded if coverage < this.
MIN_DATA_HOURS = 3

LIGHTS_ON = 9            # ZT0 hour; overridden from DB if available
BIN_LENGTH_MIN = 1       # DAM monitor bin size (minutes)
SLEEP_THRESHOLD_MIN = 5  # Consecutive inactive minutes required to count as sleep
WINDOW_HOURS = 6         # Width of each TTD window in hours
N_FOLDS = 5              # Cross-validation folds
N_ESTIMATORS = 300       # Random Forest trees per fold
RANDOM_STATE = 42        # Reproducibility seed

FEATURE_COLS = [
    'total_sleep_min',
    'n_sleep_bouts',
    'mean_bout_min',
    'longest_bout_min',
    'bouts_per_hour',
    'interruption_rate',
    'mean_wake_bout_min',
    'p_wake',
    'p_doze',
    'ZT_sin',
    'ZT_cos',
]


# ============================================================
#   HELPERS
# ============================================================

def rle(seq):
    """Run-length encoding: returns (values array, lengths array)."""
    arr = seq.values if isinstance(seq, pd.Series) else np.asarray(seq)
    if len(arr) == 0:
        return np.array([]), np.array([], dtype=int)
    changes = np.diff(arr.astype(int)) != 0
    change_indices = np.where(changes)[0] + 1
    indices = np.concatenate(([0], change_indices, [len(arr)]))
    lengths = np.diff(indices)
    values = arr[indices[:-1]]
    return values, lengths


def get_output_dir():
    """Return the analysis_results directory."""
    candidate = Path(_script_dir).parent / 'analysis' / 'analysis_results'
    if candidate.is_dir():
        return str(candidate)
    return str(Path(_script_dir) / 'analysis_results')


def get_experiment_lights_on(experiment_id):
    """Query lights_on_hour from the experiments table."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT lights_on_hour FROM experiments WHERE experiment_id = %s",
                    (experiment_id,)
                )
                row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else LIGHTS_ON
    except Exception:
        return LIGHTS_ON


# ============================================================
#   DEATH DETECTION (self-contained, no step-1 dependency)
# ============================================================

def compute_death_times_from_mt(mt_data, immobility_hours, experiment_end=None):
    """
    Detect each fly's death datetime from the loaded MT readings.

    The database only stores timestamps with non-zero activity (zero rows were
    filtered during step-1 ingest). To correctly detect immobility runs, we
    reindex each fly's time series to a complete minute grid and fill gaps with
    zero before searching for the first run of >= immobility_hours consecutive zeros.

    A fly is considered dead at the START of that first qualifying zero run.
    experiment_end defaults to the latest datetime across all flies in mt_data;
    any fly whose last non-zero reading is closer to experiment_end than
    immobility_hours cannot be confirmed dead and is excluded.

    Returns dict {fly_id: death_datetime (pd.Timestamp)}.
    """
    if experiment_end is None:
        experiment_end = mt_data['datetime'].max()

    immobility_bins = int(immobility_hours * 60)
    deaths = {}

    for fly_id, fly_df in mt_data.groupby('fly_id'):
        fly_df = fly_df.sort_values('datetime')
        data_start = fly_df['datetime'].min().round('1min')

        # Build full minute grid from data_start to experiment_end, fill zeros
        full_idx = pd.date_range(start=data_start, end=experiment_end, freq='1min')
        full_series = pd.Series(0.0, index=full_idx)

        actual = fly_df.set_index('datetime')['value']
        actual.index = actual.index.round('1min')
        # Only update timestamps that exist in our grid (avoid out-of-range assignments)
        actual = actual[actual.index.isin(full_idx)]
        full_series.update(actual)

        # Find first run of >= immobility_bins consecutive zeros
        zero_mask = (full_series.values == 0)
        vals, lens = rle(zero_mask)
        run_starts = np.concatenate(([0], np.cumsum(lens[:-1])))

        for i in range(len(vals)):
            if vals[i] and lens[i] >= immobility_bins:
                deaths[fly_id] = full_idx[run_starts[i]]
                break

    return deaths


# ============================================================
#   SLEEP FEATURE COMPUTATION (PER WINDOW)
# ============================================================

def compute_sleep_features_window(mt_values, bin_length_min=1, sleep_threshold_min=5):
    """
    Compute 9 sleep features for one 6-hour window of MT activity counts.

    Input is a complete minute-by-minute array (zeros for inactive minutes).
    Features are imputed to 0 rather than NaN where they are undefined
    (e.g. mean_bout_min=0 when there are no sleep bouts), so no rows are
    dropped downstream due to NaN features.
    """
    mt = np.asarray(mt_values, dtype=float)
    n = len(mt)

    if n == 0:
        return {k: 0.0 for k in [
            'total_sleep_min', 'n_sleep_bouts', 'mean_bout_min', 'longest_bout_min',
            'bouts_per_hour', 'interruption_rate', 'mean_wake_bout_min', 'p_wake', 'p_doze',
        ]}

    # Identify sleep: inactive runs >= threshold
    inactive = (mt == 0)
    run_vals, run_lens = rle(inactive)
    sleep_run_mask = (run_vals.astype(bool)) & (run_lens >= sleep_threshold_min)
    sleep_lens = run_lens[sleep_run_mask]
    n_bouts = int(len(sleep_lens))

    # Reconstruct per-minute sleep vector
    sleep_vec = np.zeros(n, dtype=bool)
    pos = 0
    for v, l, is_sleep in zip(run_vals, run_lens, sleep_run_mask):
        if is_sleep:
            sleep_vec[pos:pos + l] = True
        pos += l

    total_min = n * bin_length_min
    total_sleep_min = float(sleep_vec.sum() * bin_length_min)
    total_wake_min = float((~sleep_vec).sum() * bin_length_min)
    total_hours = total_min / 60.0

    # Sleep bout metrics: impute to 0 when no bouts (not NaN)
    mean_bout_min = float(np.mean(sleep_lens) * bin_length_min) if n_bouts > 0 else 0.0
    longest_bout_min = float(np.max(sleep_lens) * bin_length_min) if n_bouts > 0 else 0.0
    bouts_per_hour = n_bouts / total_hours if total_hours > 0 else 0.0

    # Interruption rate: (n_bouts-1)/n_bouts — impute to 0 when no bouts
    interruption_rate = (n_bouts - 1) / n_bouts if n_bouts > 0 else 0.0

    # Mean wake bout duration — impute to 0 when fly never woke (fully asleep window)
    wake_vals, wake_lens = rle(~sleep_vec)
    wake_bout_lens = wake_lens[wake_vals.astype(bool)]
    mean_wake_bout_min = float(np.mean(wake_bout_lens) * bin_length_min) if len(wake_bout_lens) > 0 else 0.0

    p_doze = total_sleep_min / total_min if total_min > 0 else 0.0
    p_wake = total_wake_min / total_min if total_min > 0 else 1.0

    return {
        'total_sleep_min': total_sleep_min,
        'n_sleep_bouts': float(n_bouts),
        'mean_bout_min': mean_bout_min,
        'longest_bout_min': longest_bout_min,
        'bouts_per_hour': bouts_per_hour,
        'interruption_rate': interruption_rate,
        'mean_wake_bout_min': mean_wake_bout_min,
        'p_wake': p_wake,
        'p_doze': p_doze,
    }


# ============================================================
#   WINDOW BUILDER
# ============================================================

def build_ttd_windows(
    mt_data,
    death_times,
    lights_on=LIGHTS_ON,
    bin_length_min=BIN_LENGTH_MIN,
    sleep_threshold_min=SLEEP_THRESHOLD_MIN,
    min_data_hours=MIN_DATA_HOURS,
    window_hours=WINDOW_HOURS,
):
    """
    Chop each fly's MT data into non-overlapping windows anchored backward from death.

    Window TTD=6  covers [death - 6h,  death)
    Window TTD=12 covers [death - 12h, death - 6h)  ... and so on.

    Each window is reindexed to a complete minute grid (gaps filled with 0)
    so that sleep features are computed on a full time series, not just the
    non-zero rows stored in the database.

    Windows are discarded when actual experiment coverage < min_data_hours
    (partial windows at the start of a fly's monitored lifespan).

    Returns DataFrame: fly_id, TTD, ZT_sin, ZT_cos, + 9 sleep features.
    Diagnostic counters are printed to stdout.
    """
    rows = []

    # Diagnostic counters
    n_with_windows = 0
    n_no_death = 0
    n_no_windows = 0   # had death but all windows failed coverage check

    for fly_id, fly_df in mt_data.groupby('fly_id'):
        if fly_id not in death_times:
            n_no_death += 1
            continue

        death_dt = pd.Timestamp(death_times[fly_id])
        if death_dt.tzinfo is not None:
            death_dt = death_dt.tz_localize(None)

        fly_df = fly_df.sort_values('datetime').reset_index(drop=True)
        fly_df['datetime'] = pd.to_datetime(fly_df['datetime']).dt.tz_localize(None).dt.round('1min')
        data_start = fly_df['datetime'].min()

        # Build a reindexed minute grid from data_start → death_dt
        # Missing timestamps (zero activity) are filled with 0.
        # This restores the data filtered at ingest (step 1 drops all-zero rows).
        full_idx = pd.date_range(start=data_start, end=death_dt, freq='1min')
        full_series = pd.Series(0.0, index=full_idx)
        actual = fly_df.set_index('datetime')['value']
        actual = actual[actual.index.isin(full_idx)]
        full_series.update(actual)

        n_windows_this_fly = 0
        ttd = window_hours

        while True:
            window_end = death_dt - pd.Timedelta(hours=ttd - window_hours)
            window_start = death_dt - pd.Timedelta(hours=ttd)

            # Stop if this window is entirely before the data
            if window_end <= data_start:
                break

            # Coverage check: how many hours of this window fall within actual data range
            effective_start = max(window_start, data_start)
            coverage_hours = (window_end - effective_start).total_seconds() / 3600
            if coverage_hours < min_data_hours:
                break  # partial window at lifespan start; all earlier windows worse

            # Extract window from reindexed series (guaranteed complete minute grid)
            win_mask = (full_idx >= window_start) & (full_idx < window_end)
            mt_values = full_series[win_mask].values

            # ZT at window start (continuous float, for circular encoding)
            zt_raw = (window_start.hour + window_start.minute / 60.0 - lights_on) % 24
            zt_sin = np.sin(2 * np.pi * zt_raw / 24)
            zt_cos = np.cos(2 * np.pi * zt_raw / 24)

            feats = compute_sleep_features_window(mt_values, bin_length_min, sleep_threshold_min)

            rows.append({
                'fly_id': fly_id,
                'TTD': float(ttd),
                'ZT_sin': zt_sin,
                'ZT_cos': zt_cos,
                **feats,
            })

            n_windows_this_fly += 1
            ttd += window_hours

        if n_windows_this_fly > 0:
            n_with_windows += 1
        else:
            n_no_windows += 1

    total_dead = n_with_windows + n_no_windows
    print(f"  {total_dead} flies with detected death:")
    print(f"    {n_with_windows} produced windows")
    print(f"    {n_no_windows} had no usable windows (death too close to data start)")
    print(f"  {n_no_death} flies skipped (no detected death / still alive)")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ============================================================
#   MODEL TRAINING WITH GROUP K-FOLD CV
# ============================================================

def train_rf_cv(
    windowed_features,
    n_folds=N_FOLDS,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
):
    """
    Train RF regressor with group k-fold CV (groups=fly_id, stratified by genotype).
    Returns (fold_results list, importances_df, predictions_df).
    """
    # Features are now imputed (no NaN), only drop rows missing TTD
    df = windowed_features.dropna(subset=['TTD']).reset_index(drop=True)

    # Report any remaining NaN in features (shouldn't happen after imputation)
    nan_counts = df[FEATURE_COLS].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  WARNING: NaN values in features after imputation: {nan_counts[nan_counts>0].to_dict()}")
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid windows remaining. Check data quality.")

    n_flies = df['fly_id'].nunique()
    if n_flies < n_folds:
        raise ValueError(
            f"Only {n_flies} flies with windows, but n_folds={n_folds}. "
            f"Reduce --n-folds to at most {n_flies}."
        )

    X = df[FEATURE_COLS].values
    y = df['TTD'].values
    groups = df['fly_id'].values

    # StratifiedGroupKFold: respects fly groups AND balances genotype across folds.
    # Falls back to plain GroupKFold if genotype column is absent or sklearn < 1.0.
    splits = None
    if 'genotype' in df.columns:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            from sklearn.preprocessing import LabelEncoder
            geno_labels = LabelEncoder().fit_transform(df['genotype'].astype(str))
            kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, geno_labels, groups))
        except Exception:
            pass

    if splits is None:
        kf = GroupKFold(n_splits=n_folds)
        splits = list(kf.split(X, y, groups))

    fold_results = []
    fold_importances = []
    pred_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
        })
        fold_importances.append(rf.feature_importances_)

        for true_val, pred_val in zip(y_val, y_pred):
            pred_rows.append({'actual': float(true_val), 'predicted': float(pred_val), 'fold': fold_idx + 1})

        print(f"  Fold {fold_idx + 1}/{n_folds}: MAE={mae:.2f} h, RMSE={rmse:.2f} h, R²={r2:.3f}")

    imp_matrix = np.array(fold_importances)
    importances_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance_mean': imp_matrix.mean(axis=0),
        'importance_std': imp_matrix.std(axis=0),
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

    return fold_results, importances_df, pd.DataFrame(pred_rows)


# ============================================================
#   OUTPUT FUNCTIONS
# ============================================================

def save_model_performance(fold_results, output_path):
    """Write per-fold and averaged MAE, RMSE, R² to a text file."""
    maes = [r['MAE'] for r in fold_results]
    rmses = [r['RMSE'] for r in fold_results]
    r2s = [r['R2'] for r in fold_results]

    with open(output_path, 'w') as f:
        f.write("TTD PREDICTION MODEL PERFORMANCE\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"{'Fold':<6} {'N_train':<10} {'N_val':<8} {'MAE (h)':<12} {'RMSE (h)':<12} {'R²':<8}\n")
        f.write("-" * 58 + "\n")
        for r in fold_results:
            f.write(
                f"{r['fold']:<6} {r['n_train']:<10} {r['n_val']:<8} "
                f"{r['MAE']:<12.3f} {r['RMSE']:<12.3f} {r['R2']:<8.4f}\n"
            )
        f.write("\n")
        f.write("Summary (mean ± std across folds):\n")
        f.write(f"  MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f} hours\n")
        f.write(f"  RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f} hours\n")
        f.write(f"  R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n")

    print(f"\n  Model summary:")
    print(f"    MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f} hours")
    print(f"    RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f} hours")
    print(f"    R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")


def plot_predicted_vs_actual(predictions_df, output_path):
    """Scatter plot of predicted vs actual TTD, points colored by fold."""
    fig, ax = plt.subplots(figsize=(8, 8))

    folds = sorted(predictions_df['fold'].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(folds)))

    for fold, color in zip(folds, colors):
        mask = predictions_df['fold'] == fold
        ax.scatter(
            predictions_df.loc[mask, 'actual'],
            predictions_df.loc[mask, 'predicted'],
            alpha=0.45, s=14, color=color, label=f'Fold {fold}',
        )

    all_vals = pd.concat([predictions_df['actual'], predictions_df['predicted']])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = (vmax - vmin) * 0.04
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
            'k--', linewidth=1.5, label='Perfect prediction', zorder=5)

    ax.set_xlabel('Actual TTD (hours)', fontsize=13)
    ax.set_ylabel('Predicted TTD (hours)', fontsize=13)
    ax.set_title('Predicted vs Actual Time to Death', fontsize=14)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def run_ttd_prediction(
    experiment_id=None,
    lights_on=None,
    death_immobility_hours=DEATH_IMMOBILITY_HOURS,
    min_data_hours=MIN_DATA_HOURS,
    n_folds=N_FOLDS,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
):
    """
    Main orchestrator: load data → detect deaths → build windows → train RF → save outputs.

    death_immobility_hours: consecutive inactive hours that define death (default 24).
                            Change DEATH_IMMOBILITY_HOURS at the top of this file to
                            permanently change the default.
    """
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database connection required. Check config.py and DB credentials.")

    sys.path.insert(0, _script_dir)
    from importlib import import_module
    step1 = import_module('1-prepare_data_and_health')

    # Resolve experiment
    if experiment_id is None:
        experiment_id = step1.get_latest_experiment_id()
    if not experiment_id:
        raise ValueError("No experiment found in database. Run Step 1 first.")
    print(f"\nUsing experiment_id={experiment_id}")

    # Resolve lights_on
    if lights_on is None:
        lights_on = get_experiment_lights_on(experiment_id)
    print(f"Lights-on hour (ZT0): {lights_on}:00")
    print(f"Death threshold: {death_immobility_hours}h consecutive immobility")

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # ── [1/5] Load MT readings ────────────────────────────────
    print("\n[1/5] Loading MT readings from database...")
    dam_clean = step1.load_readings_from_db(experiment_id, lights_on=lights_on)
    if dam_clean is None or len(dam_clean) == 0:
        raise ValueError(f"No readings found for experiment_id={experiment_id}.")

    mt_data = dam_clean[dam_clean['reading'] == 'MT'].copy()
    mt_data['datetime'] = pd.to_datetime(mt_data['datetime'])
    # Strip timezone for consistent comparisons
    if mt_data['datetime'].dt.tz is not None:
        mt_data['datetime'] = mt_data['datetime'].dt.tz_localize(None)
    print(f"  {len(mt_data):,} MT rows, {mt_data['fly_id'].nunique()} flies.")

    # ── [2/5] Detect death times ──────────────────────────────
    print(f"\n[2/5] Detecting death times ({death_immobility_hours}h threshold)...")
    experiment_end = mt_data['datetime'].max()
    print(f"  Experiment end reference: {experiment_end}")

    death_times = compute_death_times_from_mt(mt_data, death_immobility_hours, experiment_end)
    n_with_death = len(death_times)
    print(f"  {n_with_death}/{mt_data['fly_id'].nunique()} flies have detected deaths.")

    if n_with_death == 0:
        raise ValueError(
            f"No flies met the {death_immobility_hours}h immobility threshold. "
            "Try reducing --death-immobility-hours, or check that your experiment "
            "ran flies to death."
        )

    # ── [3/5] Build TTD-anchored windows ─────────────────────
    print(f"\n[3/5] Building TTD-anchored {WINDOW_HOURS}-hour windows...")
    windowed = build_ttd_windows(
        mt_data, death_times, lights_on,
        BIN_LENGTH_MIN, SLEEP_THRESHOLD_MIN, min_data_hours,
    )

    if windowed.empty:
        raise ValueError("No windows produced. Check data and death-detection parameters.")

    print(f"\n  Total: {len(windowed)} windows from {windowed['fly_id'].nunique()} flies.")
    print(f"  TTD range: {windowed['TTD'].min():.0f}–{windowed['TTD'].max():.0f} hours")

    # Attach genotype for fold stratification (not a model feature)
    if 'genotype' in mt_data.columns:
        fly_geno = mt_data[['fly_id', 'genotype']].drop_duplicates('fly_id')
        windowed = windowed.merge(fly_geno, on='fly_id', how='left')

    wf_path = os.path.join(output_dir, 'windowed_features.csv')
    windowed.to_csv(wf_path, index=False)
    print(f"  Saved: {wf_path}")

    # ── [4/5] Train RF with group k-fold CV ───────────────────
    print(f"\n[4/5] Training Random Forest ({n_folds}-fold group CV)...")
    fold_results, importances_df, predictions_df = train_rf_cv(
        windowed, n_folds, n_estimators, random_state
    )

    # ── [5/5] Save outputs ────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    perf_path = os.path.join(output_dir, 'model_performance.txt')
    save_model_performance(fold_results, perf_path)
    print(f"  Saved: {perf_path}")

    imp_path = os.path.join(output_dir, 'feature_importances.csv')
    importances_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    plot_path = os.path.join(output_dir, 'predicted_vs_actual.png')
    plot_predicted_vs_actual(predictions_df, plot_path)
    print(f"  Saved: {plot_path}")

    print(f"\n✅ TTD prediction complete. Outputs in: {output_dir}")
    return windowed, fold_results, importances_df


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Step 5: TTD Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 5-ttd_prediction.py
  python 5-ttd_prediction.py --experiment-id 1
  python 5-ttd_prediction.py --death-immobility-hours 24 --n-folds 5
        """
    )
    parser.add_argument('--experiment-id', type=int, default=None,
                        help='Experiment ID (default: latest)')
    parser.add_argument('--lights-on', type=int, default=None,
                        help='ZT0 hour (default: read from database)')
    parser.add_argument('--death-immobility-hours', type=int, default=DEATH_IMMOBILITY_HOURS,
                        help=f'Consecutive inactive hours defining death (default: {DEATH_IMMOBILITY_HOURS})')
    parser.add_argument('--min-data-hours', type=float, default=MIN_DATA_HOURS,
                        help=f'Min actual data coverage per window (default: {MIN_DATA_HOURS})')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help=f'CV folds (default: {N_FOLDS})')
    parser.add_argument('--n-estimators', type=int, default=N_ESTIMATORS,
                        help=f'RF trees per fold (default: {N_ESTIMATORS})')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                        help=f'Random seed (default: {RANDOM_STATE})')

    args = parser.parse_args()

    run_ttd_prediction(
        experiment_id=args.experiment_id,
        lights_on=args.lights_on,
        death_immobility_hours=args.death_immobility_hours,
        min_data_hours=args.min_data_hours,
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == '__main__':
    main()

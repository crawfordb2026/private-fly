#!/usr/bin/env python3
"""
Cluster Characterization Analysis
==================================
Characterizes which behavioral features best define each HDBSCAN cluster via
one-vs-rest Mann-Whitney U tests with Benjamini-Hochberg correction and
rank-biserial correlation effect sizes.

Inputs:
    analysis_results/umap/umap_clusters.csv  — cluster labels (cluster_hdbscan)
    PostgreSQL features_z table              — per-fly z-scored behavioral features
    analysis_results/random_forest/feature_importance.csv  — global RF importances

Outputs:
    analysis_results/cluster_characterization/cluster_feature_profiles.csv
    analysis_results/cluster_characterization/plots/cluster_{id}_top_features.png
    analysis_results/cluster_characterization/plots/cluster_feature_heatmap.png

Usage:
    python cluster_characterization.py [--experiment-id ID]
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

# Add parent directory to path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    USE_DATABASE = False
    print(f"Warning: Could not import database config: {e}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
RESULTS_DIR  = SCRIPT_DIR / 'analysis_results'
UMAP_CSV     = RESULTS_DIR / 'umap' / 'umap_clusters.csv'
RF_CSV       = RESULTS_DIR / 'random_forest' / 'feature_importance.csv'
OUT_DIR      = RESULTS_DIR / 'cluster_characterization'
PLOTS_DIR    = OUT_DIR / 'plots'

# Style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

# ---------------------------------------------------------------------------
# Display labels for features (short, plot-ready)
# ---------------------------------------------------------------------------
SHORT_LABELS = {
    "mesor_mean_z": "Mesor",
    "mesor_sd_z": "Mesor variation",
    "amplitude_mean_z": "Amplitude",
    "amplitude_sd_z": "Amplitude variation",
    "phase_mean_z": "Phase",
    "phase_sd_z": "Phase Variation",
    "periodogram_period_mean_z": "Period",
    "periodogram_period_sd_z": "Period Variation",
    "periodogram_power_mean_z": "Rhythmicity",
    "total_sleep_mean_z": "Total sleep time",
    "day_sleep_mean_z": "Daytime sleep",
    "night_sleep_mean_z": "Nighttime sleep",
    "total_bouts_mean_z": "Total number of sleep bouts",
    "day_bouts_mean_z": "Number of daytime sleep bouts",
    "night_bouts_mean_z": "Number of nighttime sleep bouts",
    "mean_bout_mean_z": "Mean sleep bout duration",
    "max_bout_mean_z": "Longest sleep bout duration",
    "mean_day_bout_mean_z": "Mean day sleep bout duration",
    "max_day_bout_mean_z": "Longest day sleep bout duration",
    "mean_night_bout_mean_z": "Mean night sleep bout duration",
    "max_night_bout_mean_z": "Longest night sleep bout duration",
    "frag_bouts_per_hour_mean_z": "Sleep bouts per hour",
    "frag_bouts_per_min_sleep_mean_z": "Sleep interruption rate",
    "mean_wake_bout_mean_z": "Wake bout duration",
    "p_wake_mean_z": "P(wake)",
    "p_doze_mean_z": "P(doze)",
    "sleep_latency_mean_z": "Sleep latency",
    "waso_mean_z": "Wake time after sleep onset (WASO)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cluster_labels():
    if not UMAP_CSV.exists():
        raise FileNotFoundError(f"Cluster labels not found: {UMAP_CSV}\nRun umap_dbscan_analysis.py first.")
    df = pd.read_csv(UMAP_CSV)
    df.columns = [c.lower() for c in df.columns]
    if 'cluster_hdbscan' not in df.columns:
        raise ValueError(f"'cluster_hdbscan' column not found in {UMAP_CSV}. Columns: {df.columns.tolist()}")
    if 'fly_id' not in df.columns:
        raise ValueError(f"'fly_id' column not found in {UMAP_CSV}.")
    # Drop noise points
    before = len(df)
    df = df[df['cluster_hdbscan'] != -1].copy()
    print(f"[Clusters] Loaded {before} flies, dropped {before - len(df)} noise (cluster=-1), {len(df)} remain.")
    print(f"  Clusters: {sorted(df['cluster_hdbscan'].unique())}")
    return df[['fly_id', 'cluster_hdbscan', 'cluster_prob']]


def load_features_from_db(experiment_id=None):
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    engine = create_engine(DATABASE_URL)
    try:
        if experiment_id is None:
            # Use the latest experiment
            exp_query = "SELECT MAX(experiment_id) AS eid FROM features_z"
            result = pd.read_sql(exp_query, engine)
            experiment_id = int(result['eid'].iloc[0])
            print(f"[DB] Using latest experiment_id: {experiment_id}")
        else:
            print(f"[DB] Using experiment_id: {experiment_id}")

        query = f"""
            SELECT fz.*, fl.genotype, fl.sex, fl.treatment
            FROM features_z fz
            JOIN flies fl ON fz.fly_id = fl.fly_id AND fz.experiment_id = fl.experiment_id
            WHERE fz.experiment_id = {experiment_id}
        """
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()

    if df is None or len(df) == 0:
        raise ValueError(f"No z-scored features found for experiment_id={experiment_id}")

    df.columns = [c.lower() for c in df.columns]
    # Drop non-feature metadata columns
    drop_cols = ['feature_id', 'experiment_id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"[DB] Loaded {len(df)} flies with {len(df.columns)} columns.")
    return df


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def rank_biserial_r(group1, group2):
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    # r = (2U / n1*n2) - 1  →  ranges [-1, 1]; positive = group1 > group2
    # Note: scipy returns U for group1; formula is r = 2*U1/(n1*n2) - 1
    r = (2.0 * u) / (n1 * n2) - 1.0
    return r


def run_cluster_characterization(merged, feature_cols, alpha=0.05):
    """
    One-vs-rest Mann-Whitney U + BH correction + rank-biserial r for every
    cluster × feature combination.

    Returns a DataFrame with columns:
        cluster, feature, u_statistic, p_value, p_adjusted,
        rank_biserial_r, direction
    """
    clusters = sorted(merged['cluster_hdbscan'].unique())
    records = []

    for clust in clusters:
        in_mask  = merged['cluster_hdbscan'] == clust
        in_flies  = merged.loc[in_mask,  feature_cols]
        out_flies = merged.loc[~in_mask, feature_cols]

        print(f"  Cluster {clust}: {in_mask.sum()} in-cluster vs {(~in_mask).sum()} out-of-cluster flies")

        p_vals   = []
        u_stats  = []
        features = []

        for feat in feature_cols:
            g1 = in_flies[feat].dropna().values
            g2 = out_flies[feat].dropna().values
            if len(g1) < 3 or len(g2) < 3:
                u_stats.append(np.nan)
                p_vals.append(np.nan)
            else:
                u, p = mannwhitneyu(g1, g2, alternative='two-sided')
                u_stats.append(u)
                p_vals.append(p)
            features.append(feat)

        # BH correction (only on non-NaN p-values)
        p_arr  = np.array(p_vals, dtype=float)
        valid  = ~np.isnan(p_arr)
        p_adj  = np.full(len(p_arr), np.nan)
        if valid.sum() > 0:
            _, p_adj[valid], _, _ = multipletests(p_arr[valid], method='fdr_bh')

        for feat, u, p, padj in zip(features, u_stats, p_vals, p_adj):
            g1 = in_flies[feat].dropna().values
            g2 = out_flies[feat].dropna().values
            r  = rank_biserial_r(g1, g2) if (len(g1) >= 3 and len(g2) >= 3) else np.nan
            direction = 'elevated' if (not np.isnan(r) and r > 0) else 'suppressed'
            records.append({
                'cluster':         clust,
                'feature':         feat,
                'u_statistic':     u,
                'p_value':         p,
                'p_adjusted':      padj,
                'rank_biserial_r': r,
                'direction':       direction,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_top_features_per_cluster(results_df, plots_dir, top_n=15, alpha=0.05):
    clusters = sorted(results_df['cluster'].unique())
    palette  = {'elevated': '#e05c5c', 'suppressed': '#4a90d9'}

    for clust in clusters:
        sub = results_df[
            (results_df['cluster'] == clust) &
            (results_df['p_adjusted'] < alpha)
        ].copy()

        if sub.empty:
            print(f"  Cluster {clust}: no significant features (p_adj < {alpha}), skipping plot.")
            continue

        sub['abs_r'] = sub['rank_biserial_r'].abs()
        sub = sub.nlargest(top_n, 'abs_r').sort_values('rank_biserial_r')

        # Map to short display labels (keep data/CSV with original names)
        sub['display'] = sub['feature'].map(lambda x: SHORT_LABELS.get(x, x))

        fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.45)))
        colors = [palette[d] for d in sub['direction']]
        ax.barh(sub['display'], sub['rank_biserial_r'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Rank-biserial correlation (r)')
        ax.set_title(f'Cluster {clust} — top {min(top_n, len(sub))} defining features\n'
                     f'(p_adj < {alpha}, one-vs-rest Mann-Whitney U)')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=palette['elevated'],   label='Elevated in cluster'),
            Patch(facecolor=palette['suppressed'], label='Suppressed in cluster'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

        plt.tight_layout()
        out_path = plots_dir / f'cluster_{clust}_top_features.png'
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  ✓ Saved: {out_path.name}")


def plot_heatmap(results_df, plots_dir, alpha=0.05):
    # Union of significant features across any cluster
    sig = results_df[results_df['p_adjusted'] < alpha]
    if sig.empty:
        print("  No significant features found for heatmap — skipping.")
        return

    sig_features = sig['feature'].unique().tolist()
    clusters     = sorted(results_df['cluster'].unique())

    pivot = (
        results_df[results_df['feature'].isin(sig_features)]
        .pivot(index='feature', columns='cluster', values='rank_biserial_r')
    )
    pivot.columns = [f'Cluster {c}' for c in pivot.columns]

    # Sort features by mean absolute r descending
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    # Create a display-indexed copy for plotting (keep original for CSVs elsewhere)
    pivot_display = pivot.copy()
    pivot_display.index = pivot_display.index.map(lambda x: SHORT_LABELS.get(x, x))

    fig_height = max(8, len(pivot) * 0.35)
    fig_width  = max(8, len(clusters) * 1.2 + 5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    hm = sns.heatmap(
        pivot_display,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=len(pivot) <= 40,   # annotate only if not too many rows
        fmt='.2f',
        linewidths=0.3,
        ax=ax,
        cbar_kws={'label': 'Rank-biserial r'},
    )
    ax.set_title(f'Cluster × Feature heatmap\n(significant features, p_adj < {alpha})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    cbar = hm.collections[0].colorbar
    cbar.set_label('Rank-biserial r', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout(pad=1.2)

    out_path = plots_dir / 'cluster_feature_heatmap.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.35)
    plt.close()
    print(f"  ✓ Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Characterize HDBSCAN clusters via one-vs-rest Mann-Whitney U tests'
    )
    parser.add_argument(
        '--experiment-id', type=int, default=None,
        help='Experiment ID (default: latest in DB)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance threshold for p_adjusted (default: 0.05)'
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('CLUSTER CHARACTERIZATION ANALYSIS')
    print('=' * 60)

    # 1. Load cluster labels
    print('\n[1/4] Loading cluster labels...')
    clusters_df = load_cluster_labels()

    # 2. Load z-scored features from DB
    print('\n[2/4] Loading z-scored features from database...')
    features_df = load_features_from_db(args.experiment_id)

    # 3. Merge on fly_id
    print('\n[3/4] Merging cluster labels with features...')
    merged = clusters_df.merge(features_df, on='fly_id', how='inner')
    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows — check that fly_id values match between "
                         "umap_clusters.csv and the features_z table.")
    print(f"  {len(merged)} flies after merge.")

    # Identify z-scored feature columns (exclude metadata)
    meta_cols   = {'fly_id', 'cluster_hdbscan', 'cluster_prob',
                   'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id'}
    feature_cols = [c for c in merged.columns if c not in meta_cols]
    print(f"  {len(feature_cols)} feature columns identified.")

    # Drop features that are entirely NaN
    all_nan = [c for c in feature_cols if merged[c].isna().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} all-NaN features.")
        feature_cols = [c for c in feature_cols if c not in all_nan]

    # 4. Run statistical tests
    print('\n[4/4] Running one-vs-rest Mann-Whitney U tests...')
    results_df = run_cluster_characterization(merged, feature_cols, alpha=args.alpha)

    # Save CSV
    out_csv = OUT_DIR / 'cluster_feature_profiles.csv'
    results_df.to_csv(out_csv, index=False)
    print(f'\n✓ Saved profiles: {out_csv}')

    n_sig = (results_df['p_adjusted'] < args.alpha).sum()
    print(f'  {n_sig} significant feature × cluster pairs (p_adj < {args.alpha})')

    # 5. Per-cluster bar charts
    print('\n[Plots] Per-cluster top-feature bar charts...')
    plot_top_features_per_cluster(results_df, PLOTS_DIR, top_n=15, alpha=args.alpha)

    # 6. Heatmap
    print('\n[Plots] Cluster × feature heatmap...')
    plot_heatmap(results_df, PLOTS_DIR, alpha=args.alpha)

    print('\n' + '=' * 60)
    print('DONE')
    print('=' * 60)
    print(f'\nOutputs in: {OUT_DIR}')
    print('  cluster_feature_profiles.csv')
    print('  plots/cluster_<id>_top_features.png  (one per cluster)')
    print('  plots/cluster_feature_heatmap.png')


if __name__ == '__main__':
    main()

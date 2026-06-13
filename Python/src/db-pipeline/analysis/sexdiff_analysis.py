#!/usr/bin/env python3
"""
Sex Difference Analysis Pipeline
=================================
Analyzes sex differences within HDBSCAN clusters.

Steps:
1. Sex composition across clusters (chi-square / Fisher exact)
2. Feature-level sex differences within each cluster
   (Mann-Whitney U + Benjamini-Hochberg FDR + rank-biserial effect size)
3. Effect size heatmap (significant features only)

Usage:
    python sexdiff_analysis.py [--experiment-id ID] [--alpha 0.05]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from importlib import import_module
import warnings
warnings.filterwarnings('ignore')

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

FEATURE_LABELS = {
    "mesor_mean_z":                    "Mesor",
    "mesor_sd_z":                      "Mesor variation",
    "amplitude_mean_z":                "Amplitude",
    "amplitude_sd_z":                  "Amplitude variation",
    "phase_mean_z":                    "Phase",
    "phase_sd_z":                      "Phase variation",
    "periodogram_period_mean_z":       "Period",
    "periodogram_period_sd_z":         "Period variation",
    "periodogram_power_mean_z":        "Rhythmicity",
    "activity_onset_zt_mean_z":        "Activity onset",
    "activity_onset_zt_sd_z":          "Activity onset variation",
    "activity_offset_zt_mean_z":       "Activity offset",
    "activity_offset_zt_sd_z":         "Activity offset variation",
    "interdaily_stability_z":          "Interdaily stability",
    "total_sleep_mean_z":              "Total sleep",
    "day_sleep_mean_z":                "Daytime sleep",
    "night_sleep_mean_z":              "Nighttime sleep",
    "total_bouts_mean_z":              "Total sleep bouts",
    "day_bouts_mean_z":                "Day sleep bouts",
    "night_bouts_mean_z":              "Night sleep bouts",
    "mean_bout_mean_z":                "Mean bout duration",
    "max_bout_mean_z":                 "Longest bout",
    "mean_day_bout_mean_z":            "Mean day bout",
    "max_day_bout_mean_z":             "Longest day bout",
    "mean_night_bout_mean_z":          "Mean night bout",
    "max_night_bout_mean_z":           "Longest night bout",
    "frag_bouts_per_hour_mean_z":      "Sleep bouts/hour",
    "frag_bouts_per_min_sleep_mean_z": "Sleep interruption rate",
    "mean_wake_bout_mean_z":           "Wake bout duration",
    "p_wake_mean_z":                   "P(wake)",
    "p_doze_mean_z":                   "P(doze)",
    "sleep_latency_mean_z":            "Sleep latency",
    "waso_mean_z":                     "WASO",
}


# ============================================================
#   DATA LOADING
# ============================================================

def load_umap_clusters(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if 'cluster_hdbscan' in df.columns and 'cluster' not in df.columns:
        df = df.rename(columns={'cluster_hdbscan': 'cluster'})
    elif 'cluster_hdbscan' in df.columns:
        df['cluster'] = df['cluster_hdbscan']
    if 'cluster' not in df.columns:
        raise ValueError(f"No cluster column found in {path}. Columns: {list(df.columns)}")
    print(f"✓ Loaded {len(df)} flies from cluster file")
    print(f"  Clusters: {sorted([c for c in df['cluster'].unique() if c != -1])}")
    print(f"  Noise: {(df['cluster'] == -1).sum()} flies")
    return df


def load_feature_data_from_db(experiment_id=None):
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    step1 = import_module('1-prepare_data_and_health')
    if experiment_id is None:
        experiment_id = step1.get_latest_experiment_id()
        print(f"[Loading] Using latest experiment_id: {experiment_id}")
    else:
        print(f"[Loading] experiment_id: {experiment_id}")
    engine = create_engine(DATABASE_URL)
    query = f"""
        SELECT fz.*, fl.genotype, fl.sex, fl.treatment, fl.monitor, fl.channel
        FROM features_z fz
        JOIN flies fl ON fz.fly_id = fl.fly_id AND fz.experiment_id = fl.experiment_id
        WHERE fz.experiment_id = {experiment_id}
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    df.columns = [c.lower() for c in df.columns]
    if 'feature_id' in df.columns:
        df = df.drop(columns=['feature_id'])
    print(f"✓ Loaded {len(df)} flies, {sum(c.endswith('_z') for c in df.columns)} z-scored features")
    return df


def subset_vehicle(df):
    df_veh = df[df['treatment'].str.upper() == 'VEH'].copy()
    print(f"✓ {len(df_veh)} vehicle flies retained")
    return df_veh


# ============================================================
#   STATISTICS HELPERS
# ============================================================

def bh_correct(p_values):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.minimum(1.0, p[order] * n / (np.arange(1, n + 1)))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    result = np.empty(n)
    result[order] = adj
    return result


def rank_biserial_r(u_stat, n1, n2):
    """
    Rank-biserial correlation from Mann-Whitney U statistic.
    Positive = group1 (females) tends to be higher.
    Range: [-1, 1].
    """
    return (2 * u_stat) / (n1 * n2) - 1


# ============================================================
#   STEP 1: SEX COMPOSITION ACROSS CLUSTERS
# ============================================================

def sex_composition_per_cluster(merged, output_dir):
    """
    Test whether sex ratios differ across HDBSCAN clusters using chi-square
    (or Fisher's exact for 2x2 tables). Run overall and per genotype.
    """
    print("\n" + "=" * 60)
    print("STEP 1: SEX COMPOSITION ACROSS CLUSTERS")
    print("=" * 60)

    data = merged[merged['cluster'] != -1].copy()
    results = []

    def run_test(label, subdf):
        ct = pd.crosstab(subdf['sex'], subdf['cluster'])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return
        if ct.shape == (2, 2):
            _, p = fisher_exact(ct.values)
            test = 'Fisher exact'
        else:
            chi2, p, dof, _ = chi2_contingency(ct.values)
            test = f'Chi-square (df={dof})'
        results.append({'group': label, 'test': test,
                        'p_value': round(p, 4), 'significant': p < 0.05})
        print(f"  {label}: {test}, p = {p:.4f}")

    print("\n[Overall contingency]")
    print(pd.crosstab(data['sex'], data['cluster']).to_string())
    run_test('ALL', data)

    print("\n[Per genotype]")
    for genotype, gdata in data.groupby('genotype'):
        run_test(genotype, gdata)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'sex_composition_tests.csv'), index=False)
    print(f"\n✓ Saved sex_composition_tests.csv")

    # Stacked bar: % sex per cluster
    cluster_sex = data.groupby(['cluster', 'sex']).size().unstack(fill_value=0)
    cluster_sex_pct = cluster_sex.div(cluster_sex.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(6, len(cluster_sex_pct) * 1.5), 5))
    cluster_sex_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.6)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Sex Composition per HDBSCAN Cluster', fontsize=13, fontweight='bold')
    ax.legend(title='Sex', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sex_composition_per_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved sex_composition_per_cluster.png")

    return results_df


# ============================================================
#   STEP 2: FEATURE-LEVEL SEX DIFFERENCES WITHIN CLUSTERS
# ============================================================

def sex_diff_within_clusters(merged, feature_cols, output_dir, alpha=0.05):
    """
    For each cluster, compare females vs males on every feature using the
    Mann-Whitney U test. Apply Benjamini-Hochberg FDR correction within
    each cluster. Report rank-biserial correlation as effect size.
    """
    print("\n" + "=" * 60)
    print("STEP 2: SEX DIFFERENCES WITHIN CLUSTERS")
    print("(Mann-Whitney U + Benjamini-Hochberg FDR)")
    print("=" * 60)

    clusters = sorted([c for c in merged['cluster'].unique() if c != -1])
    all_results = []

    for cluster in clusters:
        cdata = merged[merged['cluster'] == cluster]
        females = cdata[cdata['sex'].str.lower().str.startswith('f')]
        males   = cdata[cdata['sex'].str.lower().str.startswith('m')]

        if len(females) < 3 or len(males) < 3:
            print(f"\n  Cluster {cluster}: skipped (F={len(females)}, M={len(males)}, need ≥3 per sex)")
            continue

        print(f"\n  Cluster {cluster}: {len(females)} females, {len(males)} males")

        p_vals, u_stats, ns_f, ns_m, feats = [], [], [], [], []
        for feat in feature_cols:
            f_vals = females[feat].dropna().values
            m_vals = males[feat].dropna().values
            if len(f_vals) < 3 or len(m_vals) < 3:
                continue
            u, p = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
            feats.append(feat)
            p_vals.append(p)
            u_stats.append(u)
            ns_f.append(len(f_vals))
            ns_m.append(len(m_vals))

        if not feats:
            continue

        adj_p = bh_correct(p_vals)

        for i, feat in enumerate(feats):
            r = rank_biserial_r(u_stats[i], ns_f[i], ns_m[i])
            all_results.append({
                'cluster':          cluster,
                'feature':          feat,
                'n_female':         ns_f[i],
                'n_male':           ns_m[i],
                'u_stat':           round(u_stats[i], 1),
                'p_value':          round(p_vals[i], 4),
                'p_adj_bh':         round(float(adj_p[i]), 4),
                'rank_biserial_r':  round(r, 3),
                'direction':        'Female > Male' if r > 0 else 'Male > Female',
                'significant':      bool(adj_p[i] < alpha),
            })

        n_sig = sum(float(adj_p[i]) < alpha for i in range(len(feats)))
        print(f"    {n_sig}/{len(feats)} features significant after BH correction")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'sex_diff_within_clusters.csv'), index=False)
    print(f"\n✓ Saved sex_diff_within_clusters.csv")

    sig = results_df[results_df['significant']]
    print(f"\n  Significant feature×cluster pairs: {len(sig)}")
    if len(sig) > 0:
        print(sig[['cluster', 'feature', 'p_adj_bh', 'rank_biserial_r', 'direction']].to_string(index=False))

    return results_df


# ============================================================
#   STEP 3: EFFECT SIZE HEATMAP
# ============================================================

def plot_effect_size_heatmap(results_df, output_dir):
    """
    Heatmap of rank-biserial r for features with ≥1 significant result.
    Non-significant cells are greyed out and marked 'ns'.
    """
    print("\n[Plot] Creating effect size heatmap...")

    sig_features = results_df[results_df['significant']]['feature'].unique()
    if len(sig_features) == 0:
        print("  No significant sex differences to plot.")
        return

    plot_df = results_df[results_df['feature'].isin(sig_features)].copy()
    plot_df['label'] = plot_df['feature'].map(
        lambda x: FEATURE_LABELS.get(x, x.replace('_z', '').replace('_', ' '))
    )

    pivot_r   = plot_df.pivot_table(index='label', columns='cluster', values='rank_biserial_r')
    pivot_sig = plot_df.pivot_table(index='label', columns='cluster', values='significant').fillna(False).astype(bool)
    mask = ~pivot_sig

    fig, ax = plt.subplots(figsize=(max(7, len(pivot_r.columns) * 1.8), max(5, len(pivot_r) * 0.5)))
    sns.heatmap(
        pivot_r, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Rank-biserial r  (+  = Female higher,  −  = Male higher)'}
    )
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask.values[i, j]:
                ax.text(j + 0.5, i + 0.5, 'ns', ha='center', va='center',
                        fontsize=7, color='#999999')

    ax.set_title(
        'Sex Differences Within HDBSCAN Clusters\n'
        'Mann-Whitney U, BH FDR q < 0.05, rank-biserial effect size',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sex_diff_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved sex_diff_heatmap.png")


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Sex Difference Analysis within HDBSCAN Clusters')
    parser.add_argument('--experiment-id', type=int, default=None,
                        help='Experiment ID (default: latest)')
    parser.add_argument('--umap-clusters', type=str, default=None,
                        help='Path to umap_clusters.csv (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: analysis_results/sexdiff)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='FDR significance threshold (default: 0.05)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.umap_clusters is None:
        for candidate in [
            script_dir / 'analysis_results' / 'umap_dbscan' / 'umap_clusters.csv',
            script_dir / 'analysis_results' / 'umap' / 'umap_clusters.csv',
        ]:
            if candidate.exists():
                args.umap_clusters = str(candidate)
                break
        if args.umap_clusters is None:
            print("Error: could not find umap_clusters.csv. Pass --umap-clusters <path>")
            sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(script_dir / 'analysis_results' / 'sexdiff')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SEX DIFFERENCE ANALYSIS — HDBSCAN CLUSTERS")
    print("=" * 60)

    umap_df  = load_umap_clusters(args.umap_clusters)
    df       = load_feature_data_from_db(experiment_id=args.experiment_id)
    df_veh   = subset_vehicle(df)

    # Merge cluster labels onto VEH feature data (keep metadata from features_z)
    keep_from_clusters = [c for c in ['fly_id', 'cluster'] if c in umap_df.columns]
    merged = df_veh.merge(umap_df[keep_from_clusters], on='fly_id', how='inner')
    print(f"\n✓ Merged: {len(merged)} flies with cluster assignments")

    if len(merged) == 0:
        print("Error: no flies matched between cluster file and database. Check fly_id format.")
        sys.exit(1)

    # All z-scored feature columns
    meta_cols = {'fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel',
                 'experiment_id', 'feature_id', 'cluster', 'cluster_hdbscan',
                 'cluster_prob', 'umap1', 'umap2'}
    feature_cols = [c for c in merged.columns if c.endswith('_z') and c not in meta_cols]
    print(f"  Testing {len(feature_cols)} features")

    sex_composition_per_cluster(merged, args.output_dir)
    results_df = sex_diff_within_clusters(merged, feature_cols, args.output_dir, alpha=args.alpha)
    plot_effect_size_heatmap(results_df, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"✓ Outputs saved to: {args.output_dir}/")
    print("  - sex_composition_tests.csv")
    print("  - sex_composition_per_cluster.png")
    print("  - sex_diff_within_clusters.csv")
    print("  - sex_diff_heatmap.png")
    print("=" * 60)

    print("""
--- Methods ---
Sex composition across clusters was tested using Fisher's exact test (2×2
contingency tables) or Pearson's chi-square test of independence (larger tables;
Pearson 1900), applied overall and per genotype. Feature-level sex differences
within each HDBSCAN cluster were assessed with the two-sided Mann-Whitney U test
(Wilcoxon 1945; Mann & Whitney 1947), a non-parametric rank test that makes no
distributional assumptions and is standard for behavioural data. To control the
false discovery rate when testing all features simultaneously within each cluster,
p-values were adjusted using the Benjamini-Hochberg procedure (Benjamini &
Hochberg 1995, J. R. Stat. Soc. B 57:289-300) at q < 0.05. Effect sizes are
reported as the rank-biserial correlation r (Kerby 2014, Front. Psychol. 5:700),
derived directly from the U statistic as r = 2U/(n1*n2) - 1. This ranges from
-1 to +1; positive values indicate females are higher on that feature, negative
values indicate males are higher. Only features reaching significance in at least
one cluster appear in the heatmap; non-significant cells are marked 'ns'.
""")


if __name__ == '__main__':
    main()

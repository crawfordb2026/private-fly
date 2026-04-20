#!/usr/bin/env python3
"""
UMAP + DBSCAN Cluster Analysis Pipeline
========================================
VEH-only unsupervised clustering and pattern discovery

This script performs:
1. UMAP dimensionality reduction on PC1..PCN from pca_scores.csv (from pca_analysis.py)
2. DBSCAN clustering with automated eps detection
3. Cluster × genotype enrichment analysis
4. Cluster behavioral signatures
5. Genotype comparisons within and across clusters
6. Effect size analysis (Cliff's Delta)

Usage:
    python umap_dbscan_analysis.py [--experiment-id ID] [--output-dir analysis_results/umap]
    python umap_dbscan_analysis.py --pca-scores-csv path/to/pca_scores.csv

Run pca_analysis.py first so analysis_results/pca/pca_scores.csv exists (or pass --pca-scores-csv).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from sklearn.neighbors import NearestNeighbors
import umap
from sklearn.cluster import DBSCAN
import scikit_posthocs as sp
from importlib import import_module
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    USE_DATABASE = False
    print(f"Warning: Could not import database config: {e}")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Top 10 features for UMAP
# Note: Using lowercase _z suffix to match database convention
TOP_FEATURES = [
    "night_sleep_mean_z",
    "total_sleep_mean_z",
    "frag_bouts_per_min_sleep_mean_z",
    "max_bout_mean_z",
    "p_wake_mean_z",
    "mean_bout_mean_z",
    "max_night_bout_mean_z",
    "mean_night_bout_mean_z",
    "sleep_latency_mean_z",
    "mean_wake_bout_mean_z"
]

# UMAP input: first N principal components from pca_scores.csv (run pca_analysis.py first)
N_PCS_FOR_UMAP = 5


def load_data_from_db(experiment_id=None):
    """Load ML features Z-scored data from database."""
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    
    # Import database functions from step 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    step1 = import_module('1-prepare_data_and_health')
    
    # Use provided experiment_id, or get latest if not provided
    if experiment_id is None:
        experiment_id = step1.get_latest_experiment_id()
        if experiment_id is None:
            raise ValueError("No experiment found in database. Please specify --experiment-id")
        print(f"[Loading] Using latest experiment_id: {experiment_id}")
    else:
        print(f"[Loading] Loading experiment_id: {experiment_id}")
    
    # Load z-scored features from database
    engine = create_engine(DATABASE_URL)
    query = f"""
        SELECT 
            fz.*,
            fl.genotype,
            fl.sex,
            fl.treatment,
            fl.monitor,
            fl.channel
        FROM features_z fz
        JOIN flies fl ON fz.fly_id = fl.fly_id AND fz.experiment_id = fl.experiment_id
        WHERE fz.experiment_id = {experiment_id}
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    if df is None or len(df) == 0:
        raise ValueError(f"No z-scored features found in database for experiment_id {experiment_id}")
    
    # Normalize column names to lowercase
    df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
    
    # Remove feature_id if present
    if 'feature_id' in df.columns:
        df = df.drop(columns=['feature_id'])
    
    print(f"✓ Loaded {len(df)} flies")
    print(f"  Features: {len([c for c in df.columns if c.endswith('_z')])} z-scored features")
    print(f"  Genotypes: {sorted(df['genotype'].unique())}")
    print(f"  Sexes: {sorted(df['sex'].unique())}")
    print(f"  Treatments: {sorted(df['treatment'].unique())}")
    
    return df


def subset_vehicle(df):
    """Filter to vehicle treatment only."""
    print("\n[Subset] Filtering to vehicle (VEH) treatment...")
    df_veh = df[df['treatment'].str.upper() == 'VEH'].copy()
    print(f"✓ {len(df_veh)} flies in vehicle group")
    print(f"  Genotypes: {sorted(df_veh['genotype'].unique())}")
    print(f"  Sexes: {sorted(df_veh['sex'].unique())}")
    
    # Check for missing values
    missing = df_veh.isna().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Warning: {missing.sum()} missing values found:")
        print(missing[missing > 0])
    
    return df_veh


def prepare_umap_data_from_pca_csv(df_veh, pca_scores_path, experiment_id, n_pcs):
    """
    Load PC1..PCn from pca_scores.csv and align rows with df_veh (VEH flies in DB).

    Returns:
        umap_data: DataFrame indexed by fly_id, columns PC1..PCk
        pc_cols: list of column names used
    """
    print("\n[UMAP] Preparing data from PCA scores CSV...")
    path = Path(pca_scores_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PCA scores file not found: {path}\n"
            "Run pca_analysis.py first (same experiment) or pass --pca-scores-csv."
        )
    pca_df = pd.read_csv(path)
    pca_df.columns = [c.lower() if isinstance(c, str) else c for c in pca_df.columns]
    if 'fly_id' not in pca_df.columns:
        raise ValueError(f"PCA CSV must contain fly_id: {path}")

    if experiment_id is not None and 'experiment_id' in pca_df.columns:
        pca_df = pca_df[pca_df['experiment_id'] == experiment_id]
        print(f"  Filtered CSV to experiment_id={experiment_id}: {len(pca_df)} rows")

    if len(pca_df) == 0:
        raise ValueError("No rows left in PCA CSV after experiment filter.")

    pc_available = []
    for i in range(1, n_pcs + 1):
        # Column names are lowercased above; PCA scores use pc1, pc2, ...
        col = f'pc{i}'
        if col not in pca_df.columns:
            break
        pc_available.append(col)

    if len(pc_available) < 2:
        raise ValueError(
            f"Need at least PC1 and PC2 in CSV; found columns up to {pc_available or 'none'}."
        )

    umap_data = pca_df[pc_available].copy()
    umap_data.index = pca_df['fly_id'].astype(str)
    umap_data = umap_data.dropna()
    if len(umap_data) < len(pca_df):
        print(f"  ⚠ Removed {len(pca_df) - len(umap_data)} rows with NaN in PC columns")

    veh_ids = set(df_veh['fly_id'].astype(str))
    before = len(umap_data)
    umap_data = umap_data[umap_data.index.isin(veh_ids)]
    if len(umap_data) < before:
        print(f"  ⚠ Restricted to {len(umap_data)} flies present in VEH subset (dropped {before - len(umap_data)} not in DB VEH)")

    if len(umap_data) < 2:
        raise ValueError("Need at least 2 flies for UMAP after alignment with VEH data.")

    print(f"  Using {len(pc_available)} input dimensions: {pc_available[0]} … {pc_available[-1]} ({len(umap_data)} flies)")
    return umap_data, pc_available


def run_umap(umap_data, random_state=123):
    """Run UMAP dimensionality reduction."""
    print("\n" + "="*60)
    print("STEP 1: UMAP DIMENSIONALITY REDUCTION")
    print("="*60)
    
    print(f"\n[UMAP] Running UMAP on {len(umap_data)} flies, {umap_data.shape[1]} input dimensions...")
    print(f"  Parameters: n_neighbors=15, min_dist=0.25, metric=euclidean")
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.25,
        metric='euclidean',
        random_state=random_state
    )
    
    umap_result = reducer.fit_transform(umap_data)
    
    print(f"✓ UMAP complete")
    print(f"  Reduced from {umap_data.shape[1]}D to 2D")
    
    return umap_result


def create_umap_dataframe(umap_result, df_veh, umap_data):
    """Create UMAP DataFrame with metadata (join on fly_id)."""
    umap_df = pd.DataFrame(
        umap_result,
        columns=['UMAP1', 'UMAP2'],
        index=umap_data.index
    )
    # Ensure the index is not named 'fly_id' to avoid ambiguity later
    umap_df.index.name = None
    dfi = df_veh.set_index('fly_id').loc[umap_data.index]
    umap_df = umap_df.join(dfi[['genotype', 'sex', 'monitor', 'channel']])
    umap_df['fly_id'] = umap_df.index
    cols = ['fly_id', 'UMAP1', 'UMAP2', 'genotype', 'sex', 'monitor', 'channel']
    return umap_df[cols]


def plot_umap_genotype(umap_df, output_dir):
    """Plot UMAP colored by genotype."""
    print("\n[Plot] Creating UMAP plot (colored by genotype)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    genotypes = sorted(umap_df['genotype'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
    color_map = dict(zip(genotypes, colors))
    
    for genotype in genotypes:
        subset = umap_df[umap_df['genotype'] == genotype]
        ax.scatter(
            subset['UMAP1'], subset['UMAP2'],
            c=color_map[genotype], label=genotype,
            s=100, alpha=0.9, edgecolors='black', linewidths=0.5
        )
    
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('UMAP of Vehicle Flies (PCA scores)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title='Genotype', fontsize=14, title_fontsize=14)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'umap_genotype.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_umap_sex(umap_df, output_dir):
    """Plot UMAP shaped by sex."""
    print("\n[Plot] Creating UMAP plot (shaped by sex)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sexes = sorted(umap_df['sex'].unique())
    markers = ['o', 's', '^', 'D']
    marker_map = dict(zip(sexes, markers[:len(sexes)]))
    
    for sex in sexes:
        subset = umap_df[umap_df['sex'] == sex]
        ax.scatter(
            subset['UMAP1'], subset['UMAP2'],
            marker=marker_map[sex], label=sex,
            c='steelblue',  # Use a single color since we're distinguishing by marker
            s=100, alpha=0.9, edgecolors='black', linewidths=0.5
        )
    
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('UMAP of Vehicle Flies by Sex', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title='Sex', fontsize=14, title_fontsize=14)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'umap_sex.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_umap_genotype_sex(umap_df, output_dir):
    """Plot UMAP colored by genotype and shaped by sex."""
    print("\n[Plot] Creating UMAP plot (genotype + sex)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    genotypes = sorted(umap_df['genotype'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
    color_map = dict(zip(genotypes, colors))
    
    sexes = sorted(umap_df['sex'].unique())
    markers = ['o', 's', '^', 'D']
    marker_map = dict(zip(sexes, markers[:len(sexes)]))
    
    for genotype in genotypes:
        for sex in sexes:
            subset = umap_df[(umap_df['genotype'] == genotype) & (umap_df['sex'] == sex)]
            if len(subset) > 0:
                ax.scatter(
                    subset['UMAP1'], subset['UMAP2'],
                    c=color_map[genotype], marker=marker_map[sex],
                    s=120, alpha=0.95, label=f'{genotype} {sex}',
                    edgecolors='black', linewidths=0.5
                )
    
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('UMAP of Vehicle Flies (Genotype + Sex)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'umap_genotype_sex.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_umap_density_contours(umap_df, output_dir):
    """Plot UMAP with density contours by genotype."""
    print("\n[Plot] Creating UMAP density contours...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom colors for genotypes
    genotype_colors = {
        'Iso': '#6CC24A',
        'Rye': '#00B5D8',
        'Fmn': '#F48FB1',
        'SSS': '#C084FC'
    }
    
    # Plot points
    for genotype in sorted(umap_df['genotype'].unique()):
        subset = umap_df[umap_df['genotype'] == genotype]
        color = genotype_colors.get(genotype, 'gray')
        ax.scatter(
            subset['UMAP1'], subset['UMAP2'],
            c=color, s=60, alpha=0.7, label=genotype,
            edgecolors='black', linewidths=0.3
        )
    
    # Add density contours
    for genotype in sorted(umap_df['genotype'].unique()):
        subset = umap_df[umap_df['genotype'] == genotype]
        if len(subset) > 2:  # Need at least 3 points for contours
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([subset['UMAP1'], subset['UMAP2']])
                kde = gaussian_kde(xy)
                x_min, x_max = subset['UMAP1'].min(), subset['UMAP1'].max()
                y_min, y_max = subset['UMAP2'].min(), subset['UMAP2'].max()
                x_range = np.linspace(x_min, x_max, 50)
                y_range = np.linspace(y_min, y_max, 50)
                X, Y = np.meshgrid(x_range, y_range)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(positions).reshape(X.shape)
                color = genotype_colors.get(genotype, 'gray')
                ax.contour(X, Y, Z, colors=[color], alpha=0.6, linewidths=1.5)
            except Exception as e:
                print(f"  ⚠ Could not create contour for {genotype}: {e}")
    
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title('UMAP Density Contours by Genotype', fontsize=16, fontweight='bold')
    ax.legend(title='Genotype', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'umap_density_contours.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def find_optimal_eps(umap_xy, k=5, output_dir=None):
    """Find optimal eps parameter for DBSCAN using kNN distance plot."""
    print("\n[DBSCAN] Finding optimal eps parameter...")
    
    # Compute kNN distances
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(umap_xy)
    distances, indices = nbrs.kneighbors(umap_xy)
    
    # Get k-th nearest neighbor distances (distances[:, 0] is self, so distances[:, k] is the k-th neighbor)
    kdist = distances[:, k]  # k-th neighbor (0-indexed, so k is the k+1-th)
    kdist_sorted = np.sort(kdist)
    
    # Create kNN distance plot
    if output_dir:
        print("\n[Plot] Creating kNN distance plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(kdist_sorted, 'k-', linewidth=1.5)
        ax.scatter(range(len(kdist_sorted)), kdist_sorted, s=10, alpha=0.6)
        ax.set_xlabel('Points sorted by distance', fontsize=12)
        ax.set_ylabel(f'{k}-NN distance', fontsize=12)
        ax.set_title('kNN Distance Plot (for automated eps detection)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, 'knn_distance_plot.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {path}")
        plt.close()
    
    # Find elbow using second derivative
    if len(kdist_sorted) < 3:
        print("  ⚠ Warning: Too few points for elbow detection, using median distance")
        eps_auto = np.median(kdist_sorted)
    else:
        second_deriv = np.diff(np.diff(kdist_sorted))
        elbow_idx = np.argmax(second_deriv)
        eps_auto = kdist_sorted[elbow_idx]
    
    print(f"  Auto-detected eps: {eps_auto:.4f}")
    print(f"  Elbow point: {elbow_idx}/{len(kdist_sorted)}")
    
    return eps_auto


def run_dbscan(umap_xy, eps, k=5, random_state=123):
    """Run DBSCAN clustering on UMAP coordinates."""
    print("\n" + "="*60)
    print("STEP 2: DBSCAN CLUSTERING")
    print("="*60)
    
    print(f"\n[DBSCAN] Running DBSCAN...")
    print(f"  Parameters: eps={eps:.4f}, minPts={k+1}")
    
    db = DBSCAN(eps=eps, min_samples=k+1)
    clusters = db.fit_predict(umap_xy)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"✓ DBSCAN complete")
    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise}")
    
    return clusters


def plot_dbscan_clusters(umap_df, output_dir, filename='dbscan_clusters.png', title='DBSCAN Clusters on UMAP (Automated eps)'):
    """Plot clusters on UMAP coordinates with customizable title/filename."""
    print(f"\n[Plot] Creating cluster plot: {title}...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique clusters (excluding noise -1)
    unique_clusters = sorted([c for c in umap_df['cluster'].unique() if c != -1])
    n_clusters = len(unique_clusters)
    
    # Use Set2 colormap
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 2)))
    
    # Plot noise points first (if any)
    noise = umap_df[umap_df['cluster'] == -1]
    if len(noise) > 0:
        ax.scatter(
            noise['UMAP1'], noise['UMAP2'],
            c='gray', marker='x', s=50, alpha=0.5, label='Noise'
        )
    
    # Plot clusters
    for i, cluster in enumerate(unique_clusters):
        subset = umap_df[umap_df['cluster'] == cluster]
        ax.scatter(
            subset['UMAP1'], subset['UMAP2'],
            c=colors[i % len(colors)], label=f'Cluster {cluster}',
            s=100, alpha=0.9, edgecolors='black', linewidths=0.5
        )
    
    ax.set_xlabel('UMAP1', fontsize=14)
    ax.set_ylabel('UMAP2', fontsize=14)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title='Cluster', fontsize=14, title_fontsize=14)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_cluster_genotype_pies(umap_df, output_dir, cluster_label='HDBSCAN'):
    """Plot genotype composition pie charts for each non-noise cluster."""
    print(f"\n[Plot] Creating {cluster_label} cluster genotype pie charts...")

    # Exclude noise cluster for composition plots
    df_plot = umap_df[umap_df['cluster'] != -1].copy()
    if df_plot.empty:
        print("  ⚠ No non-noise clusters available for pie charts.")
        return

    clusters = sorted(df_plot['cluster'].unique())
    n_clusters = len(clusters)

    # Grid layout: up to 3 pie charts per row
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 7.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # Stable genotype colors across charts
    genotypes_all = sorted(df_plot['genotype'].dropna().unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(genotypes_all))))
    color_map = {g: colors[i] for i, g in enumerate(genotypes_all)}

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        sub = df_plot[df_plot['cluster'] == cluster]
        counts = sub['genotype'].value_counts().sort_index()
        total = int(counts.sum())
        pie_colors = [color_map.get(g, 'gray') for g in counts.index]
        pcts = counts.values / total * 100

        # Only annotate slices >= 5%; smaller slices get no in-slice text
        def make_autopct(pct_values):
            def autopct(pct):
                return f'{pct:.1f}%' if pct >= 5 else ''
            return autopct

        wedges, _, _ = ax.pie(
            counts.values,
            labels=None,           # labels go in legend to avoid overlap
            colors=pie_colors,
            autopct=make_autopct(pcts),
            pctdistance=0.75,
            startangle=90,
            counterclock=False,
            textprops={'fontsize': 11}
        )
        legend_labels = [f"{g} (n={counts[g]}, {pcts[j]:.1f}%)" for j, g in enumerate(counts.index)]
        ax.legend(wedges, legend_labels, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18), fontsize=10,
                  frameon=True, ncol=1)
        ax.set_title(f'Cluster {cluster} (n={total})', fontsize=18, fontweight='bold')
        ax.axis('equal')

    # Hide any unused axes
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'{cluster_label} cluster genotype composition (non-noise clusters)',
        fontsize=18,
        fontweight='bold'
    )
    plt.tight_layout()

    path = os.path.join(output_dir, 'hdbscan_cluster_genotype_pies.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def run_hdbscan(umap_xy, min_cluster_size=10, min_samples=None):
    """Run HDBSCAN clustering on UMAP coordinates."""
    print("\n" + "="*60)
    print("STEP 2: HDBSCAN CLUSTERING")
    print("="*60)
    
    print(f"\n[HDBSCAN] Running HDBSCAN...")
    print(f"  Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    clusters = hdb.fit_predict(umap_xy)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"✓ HDBSCAN complete")
    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise}")
    
    # Return clusters and probabilities if available
    probabilities = getattr(hdb, 'probabilities_', None)
    return clusters, probabilities


def cluster_genotype_enrichment(umap_df, output_dir):
    """Analyze cluster × genotype enrichment using chi-square test."""
    print("\n" + "="*60)
    print("STEP 3: CLUSTER × GENOTYPE ENRICHMENT")
    print("="*60)
    
    # Create contingency table
    cluster_geno_table = umap_df.groupby(['cluster', 'genotype']).size().reset_index(name='count')
    cluster_geno_pct = cluster_geno_table.groupby('cluster').apply(
        lambda x: x.assign(percent=100 * x['count'] / x['count'].sum())
    ).reset_index(drop=True)
    
    print("\n[Enrichment] Cluster × Genotype counts and percentages:")
    print(cluster_geno_pct.to_string(index=False))
    
    # Chi-square test
    contingency = pd.crosstab(umap_df['cluster'], umap_df['genotype'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\n[Chi-square] Test of independence:")
    print(f"  Chi² = {chi2:.4f}")
    print(f"  p-value = {p_value:.4e}")
    print(f"  df = {dof}")
    
    # Save results
    cluster_geno_pct.to_csv(
        os.path.join(output_dir, 'cluster_genotype_enrichment.csv'),
        index=False
    )
    
    enrichment_summary = pd.DataFrame({
        'chi2': [chi2],
        'p_value': [p_value],
        'df': [dof]
    })
    enrichment_summary.to_csv(
        os.path.join(output_dir, 'cluster_genotype_chi2_test.csv'),
        index=False
    )
    
    print(f"✓ Saved enrichment results")
    
    return cluster_geno_pct


def cluster_behavioral_signatures(umap_df, df_veh, top_features, output_dir):
    """Analyze behavioral signatures of each cluster."""
    print("\n" + "="*60)
    print("STEP 4: CLUSTER BEHAVIORAL SIGNATURES")
    print("="*60)
    
    # Merge cluster assignments with feature data
    # Ensure 'fly_id' is only a column label (not also an index level) on both sides
    left = umap_df[['fly_id', 'cluster']].reset_index(drop=True)
    right = df_veh.reset_index().loc[:, ['fly_id'] + top_features]
    cluster_df = left.merge(
        right,
        on='fly_id',
        how='left'
    )
    
    # Summarize features per cluster
    cluster_summary = []
    for cluster in sorted(cluster_df['cluster'].unique()):
        cluster_data = cluster_df[cluster_df['cluster'] == cluster]
        for feature in top_features:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                if len(values) > 0:
                    cluster_summary.append({
                        'cluster': cluster,
                        'feature': feature,
                        'median': values.median(),
                        'IQR': values.quantile(0.75) - values.quantile(0.25),
                        'n': len(values)
                    })
    
    cluster_summary = pd.DataFrame(cluster_summary)
    print("\n[Signatures] Cluster feature summaries:")
    print(cluster_summary.to_string(index=False))
    
    # Kruskal-Wallis tests per feature across clusters
    print("\n[Testing] Kruskal-Wallis tests (features across clusters)...")
    kw_results = []
    for feature in top_features:
        if feature in cluster_df.columns:
            groups = [group[feature].dropna().values 
                     for name, group in cluster_df.groupby('cluster')
                     if len(group[feature].dropna()) > 0]
            if len(groups) >= 2:
                try:
                    h_stat, p_value = kruskal(*groups)
                    kw_results.append({
                        'feature': feature,
                        'p_value': p_value
                    })
                except Exception as e:
                    print(f"  ⚠ Kruskal-Wallis failed for {feature}: {e}")
    
    kw_df = pd.DataFrame(kw_results)
    if len(kw_df) > 0:
        # FDR correction
        try:
            from scipy.stats import false_discovery_control
            kw_df['p_adj'] = false_discovery_control(kw_df['p_value'].values, method='bh')
        except (AttributeError, ImportError):
            print("  ⚠ FDR correction not available, using raw p-values")
            kw_df['p_adj'] = kw_df['p_value']
        
        print(kw_df.to_string(index=False))
        
        # Dunn post-hoc tests for significant features
        sig_features = kw_df[kw_df['p_adj'] < 0.05]['feature'].tolist()
        if len(sig_features) > 0:
            print(f"\n[Posthoc] Running Dunn tests for {len(sig_features)} significant features...")
            dunn_results = []
            for feature in sig_features:
                try:
                    dunn = sp.posthoc_dunn(
                        cluster_df, val_col=feature, group_col='cluster',
                        p_adjust='bonferroni'
                    )
                    # Convert to long format
                    for i in range(len(dunn)):
                        for j in range(len(dunn)):
                            if i != j:
                                dunn_results.append({
                                    'feature': feature,
                                    'comparison': f"{dunn.index[i]} vs {dunn.columns[j]}",
                                    'p_value': dunn.iloc[i, j]
                                })
                except Exception as e:
                    print(f"  ⚠ Dunn test failed for {feature}: {e}")
            
            if dunn_results:
                dunn_df = pd.DataFrame(dunn_results)
                dunn_path = os.path.join(output_dir, 'cluster_signatures_dunn_tests.csv')
                dunn_df.to_csv(dunn_path, index=False)
                print(f"✓ Saved Dunn tests: {dunn_path}")
    
    # Create heatmap
    print("\n[Plot] Creating cluster signature heatmap...")
    heatmap_mat = cluster_summary.pivot_table(
        index='cluster', columns='feature', values='median'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap_mat,
        cmap='viridis',
        center=0,
        robust=True,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': 'Median Z-Score'},
        ax=ax
    )
    ax.set_title('Cluster Behavioral Signatures (Median Z-scores)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    path = os.path.join(output_dir, 'cluster_signatures_heatmap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    # Save summary
    cluster_summary.to_csv(
        os.path.join(output_dir, 'cluster_signatures_summary.csv'),
        index=False
    )
    kw_df.to_csv(
        os.path.join(output_dir, 'cluster_signatures_kw_tests.csv'),
        index=False
    )
    
    return cluster_df, cluster_summary, kw_df


def genotype_within_clusters(umap_df, df_veh, top_features, output_dir):
    """Test genotype differences within each cluster."""
    print("\n" + "="*60)
    print("STEP 5: GENOTYPE DIFFERENCES WITHIN CLUSTERS")
    print("="*60)
    
    # Merge data
    left = umap_df[['fly_id', 'genotype', 'cluster']].reset_index(drop=True)
    right = df_veh.reset_index().loc[:, ['fly_id'] + top_features]
    cluster_df_geno = left.merge(
        right,
        on='fly_id',
        how='left'
    )
    
    clusters = sorted([c for c in cluster_df_geno['cluster'].unique() if c != -1])
    
    print(f"\n[Testing] Kruskal-Wallis tests per cluster (genotype differences)...")
    kw_results = []
    
    for cluster in clusters:
        cluster_data = cluster_df_geno[cluster_df_geno['cluster'] == cluster]
        genotypes = cluster_data['genotype'].unique()
        
        if len(genotypes) < 2:
            # Skip clusters with only one genotype
            for feature in top_features:
                kw_results.append({
                    'cluster': cluster,
                    'feature': feature,
                    'p_value': np.nan
                })
            continue
        
        for feature in top_features:
            if feature in cluster_data.columns:
                groups = [group[feature].dropna().values 
                         for name, group in cluster_data.groupby('genotype')
                         if len(group[feature].dropna()) > 0]
                if len(groups) >= 2:
                    try:
                        h_stat, p_value = kruskal(*groups)
                        kw_results.append({
                            'cluster': cluster,
                            'feature': feature,
                            'p_value': p_value
                        })
                    except Exception:
                        kw_results.append({
                            'cluster': cluster,
                            'feature': feature,
                            'p_value': np.nan
                        })
    
    kw_cluster_geno = pd.DataFrame(kw_results)
    print(kw_cluster_geno.to_string(index=False))
    
    # Significant results
    sig_kw = kw_cluster_geno[
        kw_cluster_geno['p_value'].notna() & (kw_cluster_geno['p_value'] < 0.05)
    ]
    
    if len(sig_kw) > 0:
        print(f"\n[Posthoc] Running Dunn tests for {len(sig_kw)} significant cluster-feature pairs...")
        dunn_results = []
        
        for _, row in sig_kw.iterrows():
            cluster = row['cluster']
            feature = row['feature']
            
            cluster_data = cluster_df_geno[cluster_df_geno['cluster'] == cluster]
            genotypes = cluster_data['genotype'].unique()
            
            if len(genotypes) >= 2:
                try:
                    dunn = sp.posthoc_dunn(
                        cluster_data, val_col=feature, group_col='genotype',
                        p_adjust='bonferroni'
                    )
                    # Convert to long format
                    for i in range(len(dunn)):
                        for j in range(len(dunn)):
                            if i != j and not np.isnan(dunn.iloc[i, j]):
                                dunn_results.append({
                                    'cluster': cluster,
                                    'feature': feature,
                                    'comparison': f"{dunn.index[i]} vs {dunn.columns[j]}",
                                    'p_value': dunn.iloc[i, j]
                                })
                except Exception as e:
                    print(f"  ⚠ Dunn test failed for cluster {cluster}, {feature}: {e}")
        
        if dunn_results:
            dunn_df = pd.DataFrame(dunn_results)
            dunn_path = os.path.join(output_dir, 'genotype_within_clusters_dunn.csv')
            dunn_df.to_csv(dunn_path, index=False)
            print(f"✓ Saved Dunn tests: {dunn_path}")
    
    # Create cluster × genotype × feature heatmap
    print("\n[Plot] Creating cluster × genotype × feature heatmap...")
    cluster_geno_mat = cluster_df_geno.melt(
        id_vars=['cluster', 'genotype'],
        value_vars=top_features,
        var_name='feature',
        value_name='value'
    ).groupby(['cluster', 'genotype', 'feature'])['value'].median().reset_index()
    
    # Create faceted heatmap
    clusters_to_plot = sorted([c for c in cluster_geno_mat['cluster'].unique() if c != -1])
    n_clusters = len(clusters_to_plot)
    
    fig, axes = plt.subplots(n_clusters, 1, figsize=(14, 4 * n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster in enumerate(clusters_to_plot):
        cluster_data = cluster_geno_mat[cluster_geno_mat['cluster'] == cluster]
        heatmap_data = cluster_data.pivot_table(
            index='genotype', columns='feature', values='value'
        )
        
        sns.heatmap(
            heatmap_data,
            cmap='RdBu_r',
            center=0,
            robust=True,
            square=False,
            linewidths=0.5,
            cbar_kws={'label': 'Median Z-Score'},
            ax=axes[idx]
        )
        axes[idx].set_title(f'Cluster {cluster}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature', fontsize=10)
        axes[idx].set_ylabel('Genotype', fontsize=10)
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Behavioral Signatures: Cluster × Genotype × Feature', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'cluster_genotype_feature_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    # Save results
    kw_cluster_geno.to_csv(
        os.path.join(output_dir, 'genotype_within_clusters_kw.csv'),
        index=False
    )
    
    return cluster_df_geno, kw_cluster_geno


def overall_genotype_comparisons(df_veh, top_features, output_dir):
    """Overall genotype comparisons across all flies."""
    print("\n" + "="*60)
    print("STEP 6: OVERALL GENOTYPE COMPARISONS")
    print("="*60)
    
    # Kruskal-Wallis tests
    print("\n[Testing] Kruskal-Wallis tests (genotypes across all flies)...")
    kw_results = []
    
    for feature in top_features:
        if feature in df_veh.columns:
            groups = [group[feature].dropna().values 
                     for name, group in df_veh.groupby('genotype')
                     if len(group[feature].dropna()) > 0]
            if len(groups) >= 2:
                try:
                    h_stat, p_value = kruskal(*groups)
                    kw_results.append({
                        'feature': feature,
                        'p_value': p_value
                    })
                except Exception:
                    pass
    
    kw_overall = pd.DataFrame(kw_results)
    if len(kw_overall) > 0:
        # FDR correction
        try:
            from scipy.stats import false_discovery_control
            kw_overall['FDR'] = false_discovery_control(kw_overall['p_value'].values, method='bh')
        except (AttributeError, ImportError):
            kw_overall['FDR'] = kw_overall['p_value']
        
        print(kw_overall.to_string(index=False))
        
        # Dunn post-hoc for significant features
        sig_feats = kw_overall[kw_overall['FDR'] < 0.05]['feature'].tolist()
        if len(sig_feats) > 0:
            print(f"\n[Posthoc] Running Dunn tests for {len(sig_feats)} significant features...")
            dunn_results = []
            for feature in sig_feats:
                try:
                    dunn = sp.posthoc_dunn(
                        df_veh, val_col=feature, group_col='genotype',
                        p_adjust='bonferroni'
                    )
                    for i in range(len(dunn)):
                        for j in range(len(dunn)):
                            if i != j:
                                dunn_results.append({
                                    'feature': feature,
                                    'comparison': f"{dunn.index[i]} vs {dunn.columns[j]}",
                                    'p_value': dunn.iloc[i, j]
                                })
                except Exception as e:
                    print(f"  ⚠ Dunn test failed for {feature}: {e}")
            
            if dunn_results:
                dunn_df = pd.DataFrame(dunn_results)
                dunn_path = os.path.join(output_dir, 'overall_genotype_dunn.csv')
                dunn_df.to_csv(dunn_path, index=False)
                print(f"✓ Saved Dunn tests: {dunn_path}")
        
        kw_overall.to_csv(
            os.path.join(output_dir, 'overall_genotype_kw.csv'),
            index=False
        )
    
    # Genotype phenotype heatmap
    print("\n[Plot] Creating overall genotype phenotype heatmap...")
    geno_summary = df_veh.melt(
        id_vars=['genotype'],
        value_vars=top_features,
        var_name='feature',
        value_name='value'
    ).groupby(['genotype', 'feature'])['value'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25), 'count']).reset_index()
    geno_summary.columns = ['genotype', 'feature', 'median', 'IQR', 'n']
    
    heatmap_data = geno_summary.pivot_table(
        index='genotype', columns='feature', values='median'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        robust=True,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': 'Median Z-Score'},
        ax=ax
    )
    ax.set_title('Overall Behavioral Phenotype per Genotype (VEH only)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Genotype', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    path = os.path.join(output_dir, 'overall_genotype_phenotype_heatmap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    return kw_overall


def cliffs_delta(x, y):
    """Compute Cliff's Delta effect size.
    
    Cliff's Delta: δ = P(X > Y) - P(X < Y)
    Range: [-1, 1]
    - δ = 1: All values in X are greater than all values in Y
    - δ = -1: All values in X are less than all values in Y
    - δ = 0: No dominance (equal distributions)
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan, 'negligible'
    
    # Count pairs where x > y and x < y
    count_gt = sum(1 for xi in x for yj in y if xi > yj)
    count_lt = sum(1 for xi in x for yj in y if xi < yj)
    
    # Cliff's Delta: P(X > Y) - P(X < Y)
    delta = (count_gt - count_lt) / (n_x * n_y)
    
    # Magnitude interpretation (based on Cliff 1993)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = 'negligible'
    elif abs_delta < 0.33:
        magnitude = 'small'
    elif abs_delta < 0.474:
        magnitude = 'medium'
    else:
        magnitude = 'large'
    
    return delta, magnitude


def effect_size_analysis(df_veh, top_features, output_dir):
    """Compute Cliff's Delta effect sizes for genotype pairs."""
    print("\n" + "="*60)
    print("STEP 7: EFFECT SIZE ANALYSIS (CLIFF'S DELTA)")
    print("="*60)
    
    genotypes = sorted(df_veh['genotype'].unique())
    from itertools import combinations
    pairs = list(combinations(genotypes, 2))
    
    print(f"\n[Effect Size] Computing Cliff's Delta for {len(pairs)} genotype pairs...")
    
    effect_sizes = []
    for g1, g2 in pairs:
        for feature in top_features:
            if feature in df_veh.columns:
                x = df_veh[df_veh['genotype'] == g1][feature].dropna().values
                y = df_veh[df_veh['genotype'] == g2][feature].dropna().values
                
                if len(x) > 0 and len(y) > 0:
                    delta, magnitude = cliffs_delta(x, y)
                    effect_sizes.append({
                        'comparison': f"{g1}_vs_{g2}",
                        'g1': g1,
                        'g2': g2,
                        'feature': feature,
                        'delta': delta,
                        'magnitude': magnitude
                    })
    
    effect_df = pd.DataFrame(effect_sizes)
    print(f"\n[Effect Size] Summary:")
    print(effect_df.head(20).to_string(index=False))
    
    # Create heatmap
    print("\n[Plot] Creating effect size heatmap...")
    heatmap_data = effect_df.pivot_table(
        index='comparison', columns='feature', values='delta'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': "Cliff's Delta"},
        ax=ax
    )
    ax.set_title("Effect Sizes (Cliff's Delta) Across Genotype Comparisons", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Genotype Comparison', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    path = os.path.join(output_dir, 'effect_size_cliffs_delta_heatmap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    # Save results
    effect_df.to_csv(
        os.path.join(output_dir, 'effect_size_cliffs_delta.csv'),
        index=False
    )
    
    return effect_df


def main():
    parser = argparse.ArgumentParser(
        description='UMAP + DBSCAN Analysis: VEH-only clustering and pattern discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python umap_dbscan_analysis.py
  python umap_dbscan_analysis.py --experiment-id 1
  python umap_dbscan_analysis.py --experiment-id 1 --output-dir umap_results/
  python umap_dbscan_analysis.py --pca-scores-csv analysis_results/pca/pca_scores.csv
        """
    )
    
    parser.add_argument(
        '--experiment-id',
        type=int,
        default=None,
        help='Experiment ID to use (default: latest experiment)'
    )

    parser.add_argument(
        '--pca-scores-csv',
        type=str,
        default=None,
        help='Path to pca_scores.csv from pca_analysis.py (default: analysis_results/pca/pca_scores.csv next to this script)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots and tables (default: analysis_results/umap)'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='Minimum cluster size for HDBSCAN (default: 10)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=None,
        help='Min samples for HDBSCAN (default: None → heuristic)'
    )

    # Optional DBSCAN parameters to also run DBSCAN alongside HDBSCAN
    parser.add_argument(
        '--dbscan-k',
        type=int,
        default=5,
        help='k parameter for DBSCAN eps auto-detection (default: 5)'
    )
    parser.add_argument(
        '--dbscan-eps',
        type=float,
        default=None,
        help='DBSCAN eps (default: auto-detect via kNN elbow)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        # Use analysis_results/umap within db-pipeline/analysis/ folder
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / 'analysis_results' / 'umap')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("UMAP + DBSCAN CLUSTER ANALYSIS PIPELINE")
    print("VEH-only Unsupervised Clustering and Pattern Discovery")
    print("="*60)
    
    # Load data from database
    df = load_data_from_db(experiment_id=args.experiment_id)
    experiment_id_int = int(df['experiment_id'].iloc[0])

    # Subset to vehicle
    df_veh = subset_vehicle(df)

    if len(df_veh) == 0:
        print("\n❌ Error: No vehicle flies found in dataset!")
        print("   Make sure Treatment column contains 'VEH' values")
        sys.exit(1)

    script_dir = Path(__file__).parent
    pca_scores_path = args.pca_scores_csv or str(script_dir / 'analysis_results' / 'pca' / 'pca_scores.csv')

    umap_data, pc_cols = prepare_umap_data_from_pca_csv(
        df_veh, pca_scores_path, experiment_id_int, N_PCS_FOR_UMAP
    )

    df_veh = (
        df_veh[df_veh['fly_id'].astype(str).isin(umap_data.index)]
        .set_index('fly_id')
        .loc[umap_data.index]
        .reset_index()
    )
    top_features = [f for f in TOP_FEATURES if f in df_veh.columns]
    
    # Run UMAP
    umap_result = run_umap(umap_data, random_state=123)
    umap_df = create_umap_dataframe(umap_result, df_veh, umap_data)
    
    # UMAP visualizations
    plot_umap_genotype(umap_df, args.output_dir)
    plot_umap_sex(umap_df, args.output_dir)
    plot_umap_genotype_sex(umap_df, args.output_dir)
    plot_umap_density_contours(umap_df, args.output_dir)
    
    # Run DBSCAN and HDBSCAN clustering
    umap_xy = umap_df[['UMAP1', 'UMAP2']].values

    # --- DBSCAN ---
    if args.dbscan_eps is None:
        eps_db = find_optimal_eps(umap_xy, k=args.dbscan_k, output_dir=args.output_dir)
    else:
        eps_db = args.dbscan_eps
        print(f"\n[DBSCAN] Using user-specified eps: {eps_db}")
    clusters_db = run_dbscan(umap_xy, eps_db, k=args.dbscan_k, random_state=123)

    # --- HDBSCAN ---
    clusters, probs = run_hdbscan(
        umap_xy,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    # Prepare plots (DBSCAN and HDBSCAN)
    umap_df_db = umap_df.copy()
    umap_df_db['cluster'] = clusters_db
    plot_dbscan_clusters(
        umap_df_db,
        args.output_dir,
        filename='dbscan_clusters.png',
        title='DBSCAN Clusters on UMAP (Automated eps)'
    )

    umap_df_hdb = umap_df.copy()
    umap_df_hdb['cluster'] = clusters
    plot_dbscan_clusters(
        umap_df_hdb,
        args.output_dir,
        filename='hdbscan_clusters.png',
        title='HDBSCAN Clusters on UMAP'
    )
    plot_cluster_genotype_pies(
        umap_df_hdb,
        args.output_dir,
        cluster_label='HDBSCAN'
    )
    
    # Save UMAP + cluster data
    umap_df_out = umap_df.copy()
    umap_df_out['cluster_dbscan'] = clusters_db
    umap_df_out['cluster_hdbscan'] = clusters
    if probs is not None:
        umap_df_out['cluster_prob'] = probs
    umap_df_out.to_csv(
        os.path.join(args.output_dir, 'umap_clusters.csv'),
        index=False
    )
    print(f"\n✓ Saved UMAP + cluster data: {args.output_dir}/umap_clusters.csv")
    
    # Analyses (default to HDBSCAN clusters downstream)
    umap_df = umap_df_hdb
    cluster_genotype_enrichment(umap_df, args.output_dir)
    cluster_df, cluster_summary, kw_df = cluster_behavioral_signatures(
        umap_df, df_veh, top_features, args.output_dir
    )
    cluster_df_geno, kw_cluster_geno = genotype_within_clusters(
        umap_df, df_veh, top_features, args.output_dir
    )
    kw_overall = overall_genotype_comparisons(df_veh, top_features, args.output_dir)
    effect_df = effect_size_analysis(df_veh, top_features, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - umap_clusters.csv (UMAP coordinates + cluster assignments)")
    print("  - umap_genotype.png, umap_sex.png, umap_genotype_sex.png")
    print("  - umap_density_contours.png")
    print("  - knn_distance_plot.png")
    print("  - dbscan_clusters.png")
    print("  - hdbscan_cluster_genotype_pies.png")
    print("  - cluster_genotype_enrichment.csv")
    print("  - cluster_signatures_heatmap.png")
    print("  - cluster_genotype_feature_heatmap.png")
    print("  - overall_genotype_phenotype_heatmap.png")
    print("  - effect_size_cliffs_delta_heatmap.png")
    print("  - Various CSV files with statistical test results")
    print()


if __name__ == '__main__':
    main()


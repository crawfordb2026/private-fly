#!/usr/bin/env python3
"""
PCA Analysis Pipeline
=====================
VEH-only feature analysis: PCA, Genotype Comparisons, Signature Heatmaps

This script performs:
1. PCA on z-scored features (vehicle treatment only)
2. Genotype comparisons (ANOVA/Kruskal-Wallis with automatic selection)
3. Genotype signature heatmaps (mean z-scores)
4. Effect size heatmaps (eta² or epsilon²)

Usage:
    python3 pca_analysis.py [--experiment-id ID] [--output-dir plots/]
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
from scipy.stats import shapiro, f_oneway, kruskal
from sklearn.decomposition import PCA
import scikit_posthocs as sp
from importlib import import_module
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


def run_pca(df_veh, output_dir):
    """Run PCA and create visualizations."""
    print("\n" + "="*60)
    print("STEP 1: PRINCIPAL COMPONENT ANALYSIS")
    print("="*60)
    
    # Extract z-scored features only (exclude metadata columns)
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id']
    z_cols = [col for col in df_veh.columns if col.endswith('_z') and col not in meta_cols]
    pca_features = df_veh[z_cols].copy()
    
    print(f"\n[PCA] Running PCA on {len(z_cols)} z-scored features...")
    print(f"  Features: {', '.join(z_cols[:5])}... ({len(z_cols)} total)")
    
    # Remove any remaining NaN values
    pca_features = pca_features.dropna()
    if len(pca_features) < len(df_veh):
        print(f"  ⚠ Removed {len(df_veh) - len(pca_features)} rows with NaN")
    
    # Run PCA (data is already z-scored)
    pca = PCA()
    pca_result = pca.fit_transform(pca_features)
    
    # Variance explained
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    
    print(f"\n[PCA] Variance explained:")
    print(f"  PC1: {var_exp[0]*100:.1f}%")
    print(f"  PC2: {var_exp[1]*100:.1f}%")
    print(f"  PC3: {var_exp[2]*100:.1f}%")
    print(f"  PC4: {var_exp[3]*100:.1f}%")
    print(f"  First 5 PCs: {cum_var_exp[4]*100:.1f}% cumulative")
    
    # Create PCA scores DataFrame
    pca_scores = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=pca_features.index
    )
    
    # Add metadata
    pca_scores = pca_scores.join(df_veh.loc[pca_features.index, meta_cols])
    
    # 1. Scree plot
    print("\n[Plot] Creating scree plot...")
    scree_df = pd.DataFrame({
        'PC': range(1, len(var_exp) + 1),
        'VarExp': var_exp
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(scree_df['PC'], scree_df['VarExp'], color='steelblue', alpha=0.7)
    ax.plot(scree_df['PC'], scree_df['VarExp'], 'o-', color='black', linewidth=2, markersize=6)
    ax.set_xlabel('Principal Component', fontsize=14)
    ax.set_ylabel('Proportion of Variance Explained', fontsize=14)
    ax.set_title('Scree Plot: Variance Explained', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    scree_path = os.path.join(output_dir, 'pca_scree_plot.png')
    plt.tight_layout()
    plt.savefig(scree_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {scree_path}")
    plt.close()
    
    # 2. PC1 vs PC2 scatter plot
    print("\n[Plot] Creating PC1 vs PC2 scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by genotype, shape by sex
    genotypes = sorted(pca_scores['genotype'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
    color_map = dict(zip(genotypes, colors))
    
    sexes = sorted(pca_scores['sex'].unique())
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    marker_map = dict(zip(sexes, markers[:len(sexes)]))
    
    for genotype in genotypes:
        for sex in sexes:
            subset = pca_scores[(pca_scores['genotype'] == genotype) & 
                               (pca_scores['sex'] == sex)]
            if len(subset) > 0:
                ax.scatter(
                    subset['PC1'], subset['PC2'],
                    c=[color_map[genotype]], marker=marker_map[sex],
                    s=100, alpha=0.9, label=f'{genotype} {sex}',
                    edgecolors='black', linewidths=0.5
                )
    
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=14)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=14)
    ax.set_title('PCA of Vehicle Flies (PC1 vs PC2)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
    ax.grid(True, alpha=0.3)
    
    pca_plot_path = os.path.join(output_dir, 'pca_pc1_pc2.png')
    plt.tight_layout()
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pca_plot_path}")
    plt.close()
    
    # 3. Loadings table
    print("\n[PCA] Computing loadings...")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=z_cols
    )
    
    print("\n" + "="*60)
    print("PCA LOADINGS (FIRST 4 PCs)")
    print("="*60)
    print(loadings.iloc[:, :4].round(4))
    print("="*60)
    
    # Save loadings
    loadings_path = os.path.join(output_dir, 'pca_loadings.csv')
    loadings.to_csv(loadings_path)
    print(f"\n✓ Saved loadings: {loadings_path}")

    # Save per-fly PCA scores (for UMAP or other downstream use)
    pc_cols = [c for c in pca_scores.columns if c.startswith('PC')]
    pc_cols_sorted = sorted(pc_cols, key=lambda x: int(x[2:]))
    meta_present = [c for c in meta_cols if c in pca_scores.columns]
    scores_out = pca_scores[meta_present + pc_cols_sorted]
    scores_path = os.path.join(output_dir, 'pca_scores.csv')
    scores_out.to_csv(scores_path, index=False)
    print(f"✓ Saved PCA scores: {scores_path}")

    return pca_scores, loadings, var_exp


def test_normality(grouped_data):
    """Test normality for each group using Shapiro-Wilk."""
    p_values = []
    for group_name, group_data in grouped_data:
        if len(group_data) >= 3 and len(group_data) <= 5000:  # Shapiro-Wilk limits
            stat, p = shapiro(group_data)
            p_values.append(p)
        else:
            # For very small or very large groups, assume non-normal
            p_values.append(0.0)
    return np.array(p_values)


def compute_effect_size_anova(data, groups):
    """Compute eta² (effect size for ANOVA)."""
    # Group means and grand mean
    group_means = [group_data.mean() for group_data in groups]
    grand_mean = data.mean()
    
    # Sum of squares
    ss_between = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, group_means))
    ss_total = ((data - grand_mean)**2).sum()
    
    # Eta²
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    return eta_sq


def compute_effect_size_kruskal(data, groups):
    """Compute epsilon² (effect size for Kruskal-Wallis)."""
    n = len(data)
    h_stat = kruskal(*groups).statistic
    
    # Epsilon² = (H - k + 1) / (n - k)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (n - k) if (n - k) > 0 else 0.0
    epsilon_sq = max(0.0, epsilon_sq)  # Ensure non-negative
    
    return epsilon_sq


def run_feature_test(df_veh, feature):
    """Run statistical test for a single feature."""
    # Prepare data
    data = df_veh[[feature, 'genotype']].dropna()
    if len(data) == 0:
        return None
    
    # Group by genotype
    groups = [group[feature].values for name, group in data.groupby('genotype')]
    
    if len(groups) < 2:
        return None
    
    # Test normality per group
    grouped_data = [(name, group[feature].values) for name, group in data.groupby('genotype')]
    p_values = test_normality(grouped_data)
    non_normal = np.any(p_values < 0.05)
    
    if not non_normal:
        # ANOVA path
        try:
            f_stat, p_value = f_oneway(*groups)
            
            # Validate p_value
            if not np.isfinite(p_value) or p_value < 0 or p_value > 1:
                print(f"  ⚠ Invalid p-value from ANOVA for {feature}: {p_value}, falling back to Kruskal")
                non_normal = True  # Fall back to Kruskal
            else:
                effect_size = compute_effect_size_anova(data[feature], groups)
                
                # Post-hoc: Tukey HSD
                posthoc = sp.posthoc_tukey(data, val_col=feature, group_col='genotype')
                
                return {
                    'feature': feature,
                    'test': 'ANOVA',
                    'normality': 'normal',
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'posthoc': posthoc
                }
        except Exception as e:
            print(f"  ⚠ ANOVA failed for {feature}: {e}")
            non_normal = True  # Fall back to Kruskal
    
    if non_normal:
        # Kruskal-Wallis path
        try:
            h_stat, p_value = kruskal(*groups)
            
            # Validate p_value
            if not np.isfinite(p_value) or p_value < 0 or p_value > 1:
                print(f"  ⚠ Invalid p-value from Kruskal-Wallis for {feature}: {p_value}")
                # Set to 1.0 (no significance) if invalid
                p_value = 1.0
            
            effect_size = compute_effect_size_kruskal(data[feature], groups)
            
            # Post-hoc: Dunn test
            posthoc = sp.posthoc_dunn(data, val_col=feature, group_col='genotype', p_adjust='bonferroni')
            
            return {
                'feature': feature,
                'test': 'Kruskal',
                'normality': 'non-normal',
                'p_value': p_value,
                'effect_size': effect_size,
                'posthoc': posthoc
            }
        except Exception as e:
            print(f"  ⚠ Kruskal-Wallis failed for {feature}: {e}")
            return None


def genotype_comparisons(df_veh, output_dir):
    """Run genotype comparisons for all features."""
    print("\n" + "="*60)
    print("STEP 2: GENOTYPE COMPARISONS")
    print("="*60)
    
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id']
    numeric_features = [col for col in df_veh.columns if col not in meta_cols]
    
    print(f"\n[Testing] Running statistical tests on {len(numeric_features)} features...")
    print("  (This may take a moment...)")
    
    results_list = []
    for i, feature in enumerate(numeric_features, 1):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(numeric_features)} features...")
        result = run_feature_test(df_veh, feature)
        if result:
            results_list.append(result)
    
    # Create summary table
    summary_data = []
    for res in results_list:
        summary_data.append({
            'feature': res['feature'],
            'test_used': res['test'],
            'normality': res['normality'],
            'p_value': res['p_value'],
            'effect_size': res['effect_size']
        })
    
    summary_table = pd.DataFrame(summary_data)
    
    # Validate p-values before FDR correction
    # Remove NaN, inf, or out-of-range values
    valid_mask = (
        summary_table['p_value'].notna() & 
        np.isfinite(summary_table['p_value']) &
        (summary_table['p_value'] >= 0) & 
        (summary_table['p_value'] <= 1)
    )
    
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"  ⚠ Warning: {invalid_count} features have invalid p-values (NaN, inf, or out of range)")
        print(f"    Invalid features: {summary_table[~valid_mask]['feature'].tolist()}")
        # Set invalid p-values to 1.0 (no significance)
        summary_table.loc[~valid_mask, 'p_value'] = 1.0
    
    # FDR correction using Benjamini-Hochberg method
    from scipy.stats import false_discovery_control
    p_adj = false_discovery_control(summary_table['p_value'].values, method='bh')
    summary_table['p_adj'] = p_adj
    summary_table = summary_table.sort_values('p_adj')
    
    print("\n" + "="*60)
    print("GENOTYPE FEATURE COMPARISON SUMMARY")
    print("="*60)
    print(summary_table.to_string(index=False))
    print("="*60)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'genotype_comparison_summary.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Save posthoc results (significant features only)
    sig_features = summary_table[summary_table['p_adj'] < 0.05]['feature'].tolist()
    if len(sig_features) > 0:
        print(f"\n[Posthoc] Saving posthoc results for {len(sig_features)} significant features...")
        posthoc_dir = os.path.join(output_dir, 'posthoc_results')
        os.makedirs(posthoc_dir, exist_ok=True)
        
        for res in results_list:
            if res['feature'] in sig_features:
                posthoc_path = os.path.join(posthoc_dir, f"{res['feature']}_posthoc.csv")
                res['posthoc'].to_csv(posthoc_path)
        print(f"✓ Saved posthoc results to: {posthoc_dir}/")
    
    return summary_table, results_list


def genotype_signature_heatmap(df_veh, output_dir):
    """Create genotype signature heatmap (mean z-scores)."""
    print("\n" + "="*60)
    print("STEP 3: GENOTYPE SIGNATURE HEATMAP")
    print("="*60)
    
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id']
    feature_cols = [col for col in df_veh.columns if col not in meta_cols]
    
    print(f"\n[Heatmap] Computing mean z-scores per genotype...")
    
    # Compute mean per genotype
    signature_mat = df_veh.groupby('genotype')[feature_cols].mean()
    
    print("\n[Plot] Creating genotype signature heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use seaborn for better clustering
    sns.heatmap(
        signature_mat,
        cmap='viridis',
        center=0,
        robust=True,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': 'Mean Z-Score'},
        ax=ax
    )
    
    ax.set_title('Genotype Behavioral Signatures (Mean Z Scores)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Genotype', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    heatmap_path = os.path.join(output_dir, 'genotype_signature_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {heatmap_path}")
    plt.close()
    
    # Save matrix
    matrix_path = os.path.join(output_dir, 'genotype_signature_matrix.csv')
    signature_mat.to_csv(matrix_path)
    print(f"✓ Saved matrix: {matrix_path}")
    
    return signature_mat


def effect_size_heatmap(df_veh, output_dir):
    """Create effect size heatmap."""
    print("\n" + "="*60)
    print("STEP 4: EFFECT SIZE HEATMAP")
    print("="*60)
    
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id']
    feature_cols = [col for col in df_veh.columns if col not in meta_cols]
    
    print(f"\n[Effect Size] Computing effect sizes for {len(feature_cols)} features...")
    
    effect_data = []
    for feature in feature_cols:
        data = df_veh[[feature, 'genotype']].dropna()
        if len(data) == 0:
            continue
        
        groups = [group[feature].values for name, group in data.groupby('genotype')]
        if len(groups) < 2:
            continue
        
        # Test normality
        grouped_data = [(name, group[feature].values) for name, group in data.groupby('genotype')]
        p_values = test_normality(grouped_data)
        non_normal = np.any(p_values < 0.05)
        
        if not non_normal:
            effect = compute_effect_size_anova(data[feature], groups)
        else:
            effect = compute_effect_size_kruskal(data[feature], groups)
        
        effect_data.append({
            'feature': feature,
            'effect': effect
        })
    
    effect_table = pd.DataFrame(effect_data)
    effect_table = effect_table.sort_values('effect', ascending=False)
    
    print("\n" + "="*60)
    print("RANKED FEATURES BY EFFECT SIZE")
    print("="*60)
    print(effect_table.to_string(index=False))
    print("="*60)
    
    # Create heatmap (single row)
    effect_mat = effect_table.set_index('feature')['effect'].to_frame().T
    
    print("\n[Plot] Creating effect size heatmap...")
    fig, ax = plt.subplots(figsize=(14, 2))
    
    sns.heatmap(
        effect_mat,
        cmap='viridis',
        annot=False,
        cbar_kws={'label': 'Effect Size (η² or ε²)'},
        ax=ax
    )
    
    ax.set_title('Effect Sizes Across Genotypes', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([])
    
    effect_path = os.path.join(output_dir, 'effect_size_heatmap.png')
    plt.tight_layout()
    plt.savefig(effect_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {effect_path}")
    plt.close()
    
    # Save table
    effect_table_path = os.path.join(output_dir, 'effect_size_table.csv')
    effect_table.to_csv(effect_table_path, index=False)
    print(f"✓ Saved table: {effect_table_path}")
    
    return effect_table


def main():
    parser = argparse.ArgumentParser(
        description='PCA Analysis: VEH-only feature analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pca_analysis.py
  python pca_analysis.py --experiment-id 1
  python pca_analysis.py --experiment-id 1 --output-dir analysis_results/pca
        """
    )
    
    parser.add_argument(
        '--experiment-id',
        type=int,
        default=None,
        help='Experiment ID to use (default: latest experiment)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots and tables (default: analysis_results/pca)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        # Use analysis_results/pca within db-pipeline/analysis/ folder
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / 'analysis_results' / 'pca')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("PCA ANALYSIS PIPELINE")
    print("VEH-only Feature Analysis: PCA • Genotype Comparisons • Heatmaps")
    print("="*60)
    
    # Load data from database
    df = load_data_from_db(experiment_id=args.experiment_id)
    
    # Subset to vehicle
    df_veh = subset_vehicle(df)
    
    if len(df_veh) == 0:
        print("\n❌ Error: No vehicle flies found in dataset!")
        print("   Make sure treatment column contains 'VEH' values")
        sys.exit(1)
    
    # Run analyses
    pca_scores, loadings, var_exp = run_pca(df_veh, args.output_dir)
    summary_table, results_list = genotype_comparisons(df_veh, args.output_dir)
    signature_mat = genotype_signature_heatmap(df_veh, args.output_dir)
    effect_table = effect_size_heatmap(df_veh, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - pca_scree_plot.png")
    print("  - pca_pc1_pc2.png")
    print("  - pca_loadings.csv")
    print("  - pca_scores.csv")
    print("  - genotype_comparison_summary.csv")
    print("  - genotype_signature_heatmap.png")
    print("  - genotype_signature_matrix.csv")
    print("  - effect_size_heatmap.png")
    print("  - effect_size_table.csv")
    if len(summary_table[summary_table['p_adj'] < 0.05]) > 0:
        print("  - posthoc_results/ (directory with posthoc tests)")
    print()


if __name__ == '__main__':
    main()


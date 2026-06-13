#!/usr/bin/env python3
"""
cluster_validation.py
=====================
Post-hoc validation of HDBSCAN (and/or DBSCAN) clusters from the
PCA -> UMAP -> clustering pipeline.

Prerequisites: pca_analysis.py and umap_dbscan_analysis.py must have run.

Inputs (auto-detected relative to this script):
  analysis_results/pca/pca_scores.csv      — PCA feature space (PC1..PCN)
  analysis_results/umap/umap_clusters.csv  — UMAP coords + cluster labels + genotype

Outputs saved to analysis_results/cluster_validation/:
  internal_metrics.csv              — silhouette / DB / CH per algorithm
  external_metrics.csv              — ARI / AMI / NMI per algorithm
  stability_grid.csv                — HDBSCAN param grid: n_clusters, ARI
  feature_importance_cluster_N.csv  — top features by |Cohen's d| per cluster
  feature_importance_summary.csv    — all clusters combined

Usage:
  python cluster_validation.py
  python cluster_validation.py --n-pcs 5 --top-n-features 10
"""

import argparse
import sys
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

N_PCS_DEFAULT = 5  # must match the value used in umap_dbscan_analysis.py


# ---------------------------------------------------------------------------
# 1. Internal validation (no biological labels needed)
# ---------------------------------------------------------------------------

def internal_cluster_metrics(X, labels):
    """
    Silhouette, Davies-Bouldin, Calinski-Harabasz computed in the original
    feature space (PCA scores), with noise points (label == -1) excluded.

    Interpretation:
      Silhouette [-1, 1]  — higher = better. >0.5 suggests clear separation.
      Davies-Bouldin [0,∞) — lower = better. 0 = perfectly compact, non-overlapping.
      Calinski-Harabasz [0,∞) — higher = better. No universal threshold; use for comparison.

    Returns a dict.
    """
    mask = labels != -1
    X_clean = X[mask]
    labs_clean = labels[mask]
    n_clusters = len(set(labs_clean))

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_samples_used": int(mask.sum()),
            "n_clusters": n_clusters,
            "n_noise": int((~mask).sum()),
        }

    return {
        "silhouette": round(float(silhouette_score(X_clean, labs_clean)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_clean, labs_clean)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_clean, labs_clean)), 4),
        "n_samples_used": int(mask.sum()),
        "n_clusters": n_clusters,
        "n_noise": int((~mask).sum()),
    }


# ---------------------------------------------------------------------------
# 2. External validation (requires biological labels)
# ---------------------------------------------------------------------------

def external_cluster_metrics(labels, genotype_labels):
    """
    Agreement between cluster assignments and known genotype labels.
    Noise points (label == -1) are excluded.

    Interpretation:
      ARI (Adjusted Rand Index) [-1, 1]
        Chance-corrected overlap of two label sets. 1.0 = perfect match,
        0.0 = no better than random, <0 = anti-correlated.
        ARI > 0.3 is generally considered meaningful in practice.

      AMI (Adjusted Mutual Information) [≤1]
        Information shared between the two label sets, corrected for chance.
        1.0 = perfect, 0.0 = independent. Sensitive to number of clusters.

      NMI (Normalized Mutual Information) [0, 1]
        Similar to AMI but without the chance correction, so slightly
        inflated for many-cluster solutions. Easier to compare across datasets.

    Returns a dict.
    """
    mask = np.array(labels) != -1
    labs = np.array(labels)[mask]
    geno = np.array(genotype_labels)[mask]

    if len(set(labs)) < 2:
        return {
            "ARI": np.nan,
            "AMI": np.nan,
            "NMI": np.nan,
            "n_samples_used": int(mask.sum()),
            "n_noise_excluded": int((~mask).sum()),
        }

    return {
        "ARI": round(float(adjusted_rand_score(geno, labs)), 4),
        "AMI": round(float(adjusted_mutual_info_score(geno, labs)), 4),
        "NMI": round(float(normalized_mutual_info_score(geno, labs)), 4),
        "n_samples_used": int(mask.sum()),
        "n_noise_excluded": int((~mask).sum()),
    }


# ---------------------------------------------------------------------------
# 3. Stability grid (HDBSCAN parameter sensitivity)
# ---------------------------------------------------------------------------

def stability_grid(
    umap_xy,
    genotype_labels,
    labels_baseline,
    min_cluster_sizes=(5, 10, 15, 20),
    min_samples_list=(1, 3, 5),
):
    """
    Re-run HDBSCAN over a parameter grid on the existing UMAP embedding.
    Computes ARI vs. genotype and ARI vs. the baseline clustering for each setting.

    ARI_vs_baseline near 1.0 = this setting produces the same partition as the
    default run. Wide spread -> results are parameter-sensitive.

    Returns a DataFrame sorted by min_cluster_size, min_samples.
    """
    rows = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labs = hdb.fit_predict(umap_xy)
            n_clusters = len(set(labs)) - (1 if -1 in labs else 0)
            n_noise = int((labs == -1).sum())

            mask = labs != -1
            if mask.sum() > 0 and n_clusters >= 2:
                ari_geno = round(
                    float(adjusted_rand_score(np.array(genotype_labels)[mask], labs[mask])), 4
                )
                mask_both = mask & (np.array(labels_baseline) != -1)
                ari_base = (
                    round(float(adjusted_rand_score(
                        np.array(labels_baseline)[mask_both], labs[mask_both]
                    )), 4)
                    if mask_both.sum() > 0
                    else np.nan
                )
            else:
                ari_geno = np.nan
                ari_base = np.nan

            rows.append(
                {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "ARI_vs_genotype": ari_geno,
                    "ARI_vs_baseline": ari_base,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Feature importance: Cohen's d (cluster vs. rest)
# ---------------------------------------------------------------------------

def _cohens_d(group, rest):
    """Signed Cohen's d: (mean_cluster - mean_rest) / pooled_std."""
    n1, n2 = len(group), len(rest)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * group.std(ddof=1) ** 2 + (n2 - 1) * rest.std(ddof=1) ** 2) / (
        n1 + n2 - 2
    )
    pooled_std = np.sqrt(pooled_var)
    return 0.0 if pooled_std == 0 else (group.mean() - rest.mean()) / pooled_std


def cluster_feature_importance(X_df, labels, top_n=10):
    """
    For each non-noise cluster, rank features by |Cohen's d| (cluster vs. all
    other non-noise flies). Positive d = feature is elevated in this cluster.

    Returns dict: {cluster_id: DataFrame of top_n features}.
    """
    mask_all = labels != -1
    X_vals = X_df.values[mask_all]
    labs_nn = labels[mask_all]
    feat_names = list(X_df.columns)

    results = {}
    for cluster_id in sorted(set(labs_nn)):
        in_cl = labs_nn == cluster_id
        rows = []
        for i, col in enumerate(feat_names):
            g = X_vals[in_cl, i]
            r = X_vals[~in_cl, i]
            g = g[~np.isnan(g)]
            r = r[~np.isnan(r)]
            d = _cohens_d(pd.Series(g), pd.Series(r))
            rows.append(
                {
                    "feature": col,
                    "cohens_d": round(float(d), 4) if not np.isnan(d) else np.nan,
                    "abs_d": abs(float(d)) if not np.isnan(d) else np.nan,
                    "mean_cluster": round(float(g.mean()), 4) if len(g) > 0 else np.nan,
                    "mean_rest": round(float(r.mean()), 4) if len(r) > 0 else np.nan,
                    "n_cluster": len(g),
                    "n_rest": len(r),
                }
            )
        df_imp = (
            pd.DataFrame(rows)
            .sort_values("abs_d", ascending=False)
            .drop(columns="abs_d")
            .head(top_n)
            .reset_index(drop=True)
        )
        results[cluster_id] = df_imp

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    default_umap_csv = script_dir / "analysis_results" / "umap" / "umap_clusters.csv"
    default_pca_csv = script_dir / "analysis_results" / "pca" / "pca_scores.csv"
    default_out = script_dir / "analysis_results" / "cluster_validation"

    parser = argparse.ArgumentParser(description="Cluster validation pipeline")
    parser.add_argument("--umap-csv", default=str(default_umap_csv))
    parser.add_argument("--pca-csv", default=str(default_pca_csv))
    parser.add_argument("--output-dir", default=str(default_out))
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=N_PCS_DEFAULT,
        help="Number of PCs used as UMAP input (default: 5)",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=10,
        help="Top N features by |Cohen's d| to report per cluster (default: 10)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and align inputs
    # -----------------------------------------------------------------------
    for p in [args.umap_csv, args.pca_csv]:
        if not Path(p).is_file():
            sys.exit(f"ERROR: required input not found:\n  {p}\n"
                     "Run pca_analysis.py and umap_dbscan_analysis.py first.")

    umap_df = pd.read_csv(args.umap_csv)
    umap_df.columns = [c.lower() for c in umap_df.columns]
    umap_df["fly_id"] = umap_df["fly_id"].astype(str)

    pca_df = pd.read_csv(args.pca_csv)
    pca_df.columns = [c.lower() for c in pca_df.columns]
    pca_df["fly_id"] = pca_df["fly_id"].astype(str)

    # Detect PC columns in pca_scores.csv
    pc_cols_all = sorted(
        [c for c in pca_df.columns if c.startswith("pc") and c[2:].isdigit()],
        key=lambda c: int(c[2:]),
    )
    pc_cols_use = pc_cols_all[: args.n_pcs]
    if len(pc_cols_use) < 2:
        sys.exit(f"ERROR: need at least 2 PC columns in {args.pca_csv}; "
                 f"found: {pc_cols_all}")

    merged = umap_df.merge(pca_df[["fly_id"] + pc_cols_use], on="fly_id", how="inner")
    if len(merged) == 0:
        sys.exit("ERROR: no overlapping fly_ids between umap_clusters.csv and pca_scores.csv")

    print(f"[Load] {len(merged)} flies matched  |  PC features used: {pc_cols_use}")

    X = merged[pc_cols_use].values.astype(float)
    X_df = pd.DataFrame(X, columns=pc_cols_use)

    genotype_labels = merged["genotype"].values

    # UMAP embedding already stored in the CSV
    if "umap1" not in merged.columns or "umap2" not in merged.columns:
        sys.exit("ERROR: umap_clusters.csv must contain umap1 and umap2 columns")
    umap_xy = merged[["umap1", "umap2"]].values.astype(float)

    # Detect which cluster columns exist (support both and legacy "cluster")
    has_hdbscan = "cluster_hdbscan" in merged.columns
    has_dbscan = "cluster_dbscan" in merged.columns
    # legacy fallback: a single "cluster" column -> treat as hdbscan
    if not has_hdbscan and not has_dbscan and "cluster" in merged.columns:
        merged["cluster_hdbscan"] = merged["cluster"]
        has_hdbscan = True
        print("[Warn] Only 'cluster' column found; treating it as HDBSCAN labels.")

    if not has_hdbscan and not has_dbscan:
        sys.exit("ERROR: umap_clusters.csv needs cluster_hdbscan and/or cluster_dbscan columns")

    labels_hdb = merged["cluster_hdbscan"].values.astype(int) if has_hdbscan else None
    labels_db = merged["cluster_dbscan"].values.astype(int) if has_dbscan else None

    # -----------------------------------------------------------------------
    # 1. Internal metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: INTERNAL CLUSTER VALIDATION (PCA feature space)")
    print("=" * 60)
    print("(Noise points excluded. Higher silhouette / lower DB / higher CH = better.)")

    internal_rows = []
    for alg, labels in [("HDBSCAN", labels_hdb), ("DBSCAN", labels_db)]:
        if labels is None:
            continue
        m = internal_cluster_metrics(X, labels)
        m["algorithm"] = alg
        internal_rows.append(m)
        print(f"\n  {alg}:  n_clusters={m['n_clusters']}  noise={m['n_noise']}  "
              f"silhouette={m['silhouette']}  davies_bouldin={m['davies_bouldin']}  "
              f"calinski_harabasz={m['calinski_harabasz']}")

    internal_df = pd.DataFrame(internal_rows)[
        ["algorithm", "n_clusters", "n_samples_used", "n_noise",
         "silhouette", "davies_bouldin", "calinski_harabasz"]
    ]
    p = out_dir / "internal_metrics.csv"
    internal_df.to_csv(p, index=False)
    print(f"\n[OK] Saved: {p}")

    # -----------------------------------------------------------------------
    # 2. External metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: EXTERNAL VALIDATION (cluster labels vs. genotype)")
    print("=" * 60)
    print("(ARI/AMI/NMI > 0 = clusters share structure with genotypes.)")

    external_rows = []
    for alg, labels in [("HDBSCAN", labels_hdb), ("DBSCAN", labels_db)]:
        if labels is None:
            continue
        m = external_cluster_metrics(labels, genotype_labels)
        m["algorithm"] = alg
        external_rows.append(m)
        print(f"\n  {alg}:  ARI={m['ARI']}  AMI={m['AMI']}  NMI={m['NMI']}  "
              f"(n={m['n_samples_used']}, excluded noise={m['n_noise_excluded']})")

    external_df = pd.DataFrame(external_rows)[
        ["algorithm", "n_samples_used", "n_noise_excluded", "ARI", "AMI", "NMI"]
    ]
    p = out_dir / "external_metrics.csv"
    external_df.to_csv(p, index=False)
    print(f"\n[OK] Saved: {p}")

    # -----------------------------------------------------------------------
    # 3. Stability grid (HDBSCAN only — re-runs on existing UMAP embedding)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: HDBSCAN STABILITY GRID")
    print("=" * 60)

    if has_hdbscan:
        stab_df = stability_grid(
            umap_xy,
            genotype_labels,
            labels_hdb,
            min_cluster_sizes=(5, 10, 15, 20),
            min_samples_list=(1, 3, 5),
        )
        p = out_dir / "stability_grid.csv"
        stab_df.to_csv(p, index=False)

        print("\nResults (sorted by ARI_vs_genotype):")
        print(stab_df.sort_values("ARI_vs_genotype", ascending=False).to_string(index=False))

        ari_vals = stab_df["ARI_vs_genotype"].dropna()
        if len(ari_vals) > 1:
            spread = ari_vals.std()
            print(f"\nRobustness: ARI range [{ari_vals.min():.3f}, {ari_vals.max():.3f}]"
                  f"  std={spread:.3f}")
            if spread < 0.05:
                print("  -> Low variance: cluster structure is ROBUST to parameter choice.")
            elif spread < 0.15:
                print("  -> Moderate variance: some sensitivity — review grid for preferred params.")
            else:
                print("  -> High variance: results are PARAMETER-SENSITIVE. Interpret with caution.")

        print(f"\n[OK] Saved: {p}")
    else:
        print("  Skipped — no HDBSCAN labels found in umap_clusters.csv.")

    # -----------------------------------------------------------------------
    # 4. Feature importance: Cohen's d per cluster vs. rest
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE IMPORTANCE (Cohen's d, cluster vs. rest)")
    print("=" * 60)

    # Prefer HDBSCAN; fall back to DBSCAN
    if has_hdbscan:
        target_labels, alg_name = labels_hdb, "HDBSCAN"
    else:
        target_labels, alg_name = labels_db, "DBSCAN"

    print(f"  Using {alg_name} labels.  Top {args.top_n_features} features per cluster.")
    print("  Positive Cohen's d = feature is ELEVATED vs. rest of non-noise flies.")

    importance = cluster_feature_importance(X_df, target_labels, top_n=args.top_n_features)

    summary_rows = []
    for cid, df_imp in importance.items():
        n_in = int((target_labels == cid).sum())
        print(f"\n  Cluster {cid}  (n={n_in} flies):")
        print(df_imp[["feature", "cohens_d", "mean_cluster", "mean_rest"]].to_string(index=False))
        p = out_dir / f"feature_importance_cluster_{cid}.csv"
        df_imp.to_csv(p, index=False)
        print(f"  [OK] Saved: {p}")
        for rank, row in enumerate(df_imp.itertuples(index=False), start=1):
            summary_rows.append(
                {
                    "cluster": cid,
                    "rank": rank,
                    "feature": row.feature,
                    "cohens_d": row.cohens_d,
                    "mean_cluster": row.mean_cluster,
                    "mean_rest": row.mean_rest,
                    "n_cluster": row.n_cluster,
                    "n_rest": row.n_rest,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    p = out_dir / "feature_importance_summary.csv"
    summary_df.to_csv(p, index=False)
    print(f"\n[OK] Saved combined summary: {p}")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLUSTER VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs in: {out_dir}/")
    print("  internal_metrics.csv             — silhouette / Davies-Bouldin / Calinski-Harabasz")
    print("  external_metrics.csv             — ARI / AMI / NMI vs. genotype")
    print("  stability_grid.csv               — HDBSCAN param grid ARI sensitivity")
    print("  feature_importance_cluster_N.csv — top features per cluster")
    print("  feature_importance_summary.csv   — all clusters combined")


if __name__ == "__main__":
    main()

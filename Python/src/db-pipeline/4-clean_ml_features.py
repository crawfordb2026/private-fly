#!/usr/bin/env python3
"""
Pipeline Step 4: Clean ML Features

This script:
1. Reads ML features from database (Step 3)
2. Removes flies with problematic feature values:
   - Zero total sleep
   - Zero sleep bouts
   - Zero/NaN P_doze
3. Removes IQR outliers (per group) for total_sleep_mean
4. Fixes NaN values (replaces with 0 or group mean)
5. Creates z-scored feature table
6. Saves cleaned and z-scored versions

Output: Saved to database (features and features_z tables)

This step prepares the feature table for machine learning by removing
problematic flies and normalizing features.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from importlib import import_module

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    import psycopg2
    from psycopg2.extras import execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#   USER CONFIGURATION
# ============================================================


# IQR outlier detection settings
DEFAULT_IQR_MULTIPLIER = 1.5  # Standard IQR multiplier for outlier detection


# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def compute_iqr_bounds(df, column='total_sleep_mean', multiplier=1.5):
    """
    Compute IQR bounds for outlier detection.
    
    Args:
        df: DataFrame with the column to analyze
        column: Column name to compute IQR for
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        dict with Q1, Q3, IQR, lower_bound, upper_bound
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': Q1 - multiplier * IQR,
        'upper_bound': Q3 + multiplier * IQR
    }


# ============================================================
#   MAIN CLEANING FUNCTIONS
# ============================================================

def remove_problematic_flies(ML_features):
    """
    Remove flies with zero sleep or zero bouts.
    Note: p_doze_mean can be NaN if there are no wake-to-sleep transitions, which is valid.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        tuple: (cleaned_df, removed_flies_df)
    """
    df = ML_features.copy()
    
    # Check that required columns exist
    required_cols = ['total_sleep_mean', 'total_bouts_mean']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for problematic fly detection: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Identify problematic flies (only zero sleep or zero bouts)
    # Note: p_doze_mean can be NaN legitimately, so we don't remove based on that
    df['never_slept'] = (df['total_sleep_mean'] == 0) | df['total_sleep_mean'].isna()
    df['zero_sleep_bouts'] = (df['total_bouts_mean'] == 0) | df['total_bouts_mean'].isna()
    
    # Count problematic flies by reason
    never_slept_count = df['never_slept'].sum()
    zero_bouts_count = df['zero_sleep_bouts'].sum()
    
    # Find flies to remove (only those with zero sleep OR zero bouts)
    problematic_mask = df['never_slept'] | df['zero_sleep_bouts']
    removed_flies = df[problematic_mask].copy()
    
    # Select relevant columns for report
    report_cols = ['fly_id', 'genotype', 'sex', 'treatment',
                   'total_sleep_mean', 'total_bouts_mean',
                   'never_slept', 'zero_sleep_bouts']
    if 'p_doze_mean' in removed_flies.columns:
        report_cols.append('p_doze_mean')
    
    removed_flies_report = removed_flies[report_cols].sort_values(['genotype', 'sex', 'treatment', 'fly_id'])
    
    # Remove problematic flies
    df_clean = df[~problematic_mask].copy()
    
    # Drop helper columns
    df_clean = df_clean.drop(columns=['never_slept', 'zero_sleep_bouts'])
    
    return df_clean, removed_flies_report


def remove_iqr_outliers(ML_features, column='total_sleep_mean', multiplier=1.5):
    """
    Remove IQR outliers per group (genotype × sex × treatment).
    
    Args:
        ML_features: Input feature DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        tuple: (cleaned_df, iqr_bounds_df, outlier_summary_df)
    """
    df = ML_features.copy()
    
    # Check that required columns exist
    required_cols = ['genotype', 'sex', 'treatment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for IQR calculation: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Check for NaN values in grouping columns
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: Found NaN values in grouping columns: {nan_counts.to_dict()}")
        # Drop rows with NaN in grouping columns
        df = df.dropna(subset=required_cols)
        if len(df) == 0:
            raise ValueError("No rows remaining after removing NaN values from genotype, sex, treatment columns")
    
    # Compute IQR bounds per group
    iqr_results = []
    for (genotype, sex, treatment), group in df.groupby(required_cols):
        if len(group) == 0:
            continue
        bounds = compute_iqr_bounds(group, column, multiplier)
        bounds['genotype'] = genotype
        bounds['sex'] = sex
        bounds['treatment'] = treatment
        iqr_results.append(bounds)
    
    if len(iqr_results) == 0:
        raise ValueError("No groups found for IQR calculation. Check that genotype, sex, and treatment columns exist and have valid values.")
    
    iqr_bounds = pd.DataFrame(iqr_results)
    
    # Ensure iqr_bounds has the merge columns
    required_cols = ['genotype', 'sex', 'treatment']
    if not all(col in iqr_bounds.columns for col in required_cols):
        raise ValueError(f"iqr_bounds missing required columns. Expected: {required_cols}, Has: {list(iqr_bounds.columns)}")
    
    # Verify df also has the merge columns
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df missing required columns for merge. Expected: {required_cols}, Has: {list(df.columns)}")
    
    # Merge bounds back to main dataframe
    df = df.merge(iqr_bounds, on=['genotype', 'sex', 'treatment'], how='left')
    
    # Identify outliers
    df['is_outlier_IQR_total_sleep'] = (
        (df[column] < df['lower_bound']) |
        (df[column] > df['upper_bound'])
    )
    
    # Create summary before removal
    outlier_summary = df.groupby(['genotype', 'sex', 'treatment']).agg({
        'fly_id': 'count',
        'is_outlier_IQR_total_sleep': 'sum'
    }).reset_index()
    outlier_summary.columns = ['genotype', 'sex', 'treatment', 'n_total', 'n_outliers']
    outlier_summary['percent_outliers'] = (outlier_summary['n_outliers'] / outlier_summary['n_total'] * 100).round(1)
    
    # Remove outliers
    df_clean = df[~df['is_outlier_IQR_total_sleep']].copy()
    
    # Drop IQR calculation columns
    qc_cols = ['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound', 'is_outlier_IQR_total_sleep']
    df_clean = df_clean.drop(columns=[col for col in qc_cols if col in df_clean.columns])
    
    return df_clean, iqr_bounds, outlier_summary


def fix_nan_values(ML_features):
    """
    Fix NaN values by replacing with 0 or group mean.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        DataFrame with NaN values fixed
    """
    df = ML_features.copy()
    
    # Columns that should be 0 when NaN (bout metrics when no sleep)
    zero_cols = [
        'day_bouts_mean', 'night_bouts_mean',
        'mean_day_bout_mean', 'max_day_bout_mean',
        'mean_night_bout_mean', 'max_night_bout_mean',
        'frag_bouts_per_min_sleep_mean'
    ]
    
    # Columns that should use group mean when NaN
    mean_cols = [
        'sleep_latency_mean', 'WASO_mean',
        'Mesor_sd', 'Amp_sd', 'Phase_sd'
    ]
    
    # Replace NaN with 0 for zero_cols
    for col in zero_cols:
        if col in df.columns:
            # fillna handles both pandas NA and numpy NaN
            df[col] = df[col].fillna(0)
    
    # Replace NaN with group mean for mean_cols
    for col in mean_cols:
        if col in df.columns:
            # Compute group means (returns Series aligned with df)
            group_means = df.groupby(['genotype', 'sex', 'treatment'])[col].transform('mean')
            # fillna handles both pandas NA and numpy NaN
            df[col] = df[col].fillna(group_means)
    
    return df


def create_z_scored_features(ML_features):
    """
    Create z-scored (standardized) feature table.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        DataFrame with z-scored features
    """
    df = ML_features.copy()
    
    # Metadata columns to keep (exclude feature_id and other non-feature columns)
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'feature_id', 'experiment_id']
    
    # Get numeric columns (exclude metadata and non-feature columns)
    numeric_cols = [col for col in df.columns 
                   if col not in meta_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Create z-scored version
    df_z = df[meta_cols].copy()
    
    for col in numeric_cols:
        z_col = f'{col}_z'  # Use lowercase _z to match database schema
        # Z-score: (x - mean) / std
        df_z[z_col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Remove n_days_z if it exists
    if 'n_days_z' in df_z.columns:
        df_z = df_z.drop(columns=['n_days_z'])
    
    return df_z


def run_diagnostics(ML_features_clean):
    """
    Run diagnostic checks on cleaned features.
    
    Args:
        ML_features_clean: Cleaned feature DataFrame
    
    Returns:
        dict with diagnostic results
    """
    df = ML_features_clean.copy()
    
    diagnostics = {}
    
    # 1. Check for NA
    na_summary = df.isna().sum()
    na_summary = na_summary[na_summary > 0]
    diagnostics['NA_summary'] = na_summary.to_dict() if len(na_summary) > 0 else {}
    
    # 2. Check for NaN
    nan_summary = {}
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        nan_count = pd.isna(df[col]).sum()
        if nan_count > 0:
            nan_summary[col] = int(nan_count)
    diagnostics['NaN_summary'] = nan_summary
    
    # 3. List flies with any NaN
    numeric_df = df.select_dtypes(include=[np.number])
    nan_flies = df[pd.isna(numeric_df).any(axis=1)]
    diagnostics['flies_with_NaN'] = nan_flies[['fly_id', 'genotype', 'sex', 'treatment']].copy() if len(nan_flies) > 0 else pd.DataFrame()
    
    # 4. Which columns have NaN
    nan_columns = []
    for col in numeric_cols:
        if pd.isna(df[col]).any():
            nan_columns.append(col)
    diagnostics['NaN_columns'] = nan_columns
    
    # 5. Diagnose fragmentation metrics
    frag_cols = ['frag_bouts_per_hour_mean', 'frag_bouts_per_min_sleep_mean']
    frag_nan = df[pd.isna(df[frag_cols]).any(axis=1)]
    diagnostics['frag_problems'] = frag_nan[['fly_id', 'genotype', 'sex', 'treatment'] + frag_cols].copy() if len(frag_nan) > 0 else pd.DataFrame()
    
    # 6. Diagnose sleep bout structure
    sleep_struct = df[
        (df['total_sleep_mean'] == 0) | (df['total_bouts_mean'] == 0)
    ][['fly_id', 'genotype', 'sex', 'treatment', 'total_sleep_mean', 'total_bouts_mean']].copy()
    diagnostics['sleep_structure_problems'] = sleep_struct
    
    return diagnostics


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def clean_ml_features(
    iqr_multiplier=DEFAULT_IQR_MULTIPLIER,
    save_diagnostics=True,
    experiment_id=None
):
    """
    Main function to clean ML features.
    
    Args:
        iqr_multiplier: IQR multiplier for outlier detection
        save_diagnostics: Whether to save diagnostic reports
        experiment_id: Experiment ID to use (None = use latest)
    
    Returns:
        tuple: (cleaned_df, z_scored_df)
    """
    # Require database
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    
    # ============================================================
    # STEP 1: Load ML features from database
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    # Import database functions from step 1
    step1 = import_module('1-prepare_data_and_health')
    
    # Use provided experiment_id, or get latest if not provided
    experiment_id_param = experiment_id
    if experiment_id_param is None:
        experiment_id = step1.get_latest_experiment_id()
    else:
        experiment_id = experiment_id_param
    
    if not experiment_id:
        raise ValueError("No experiment found in database")
    
    # Load features from database (join with flies table to get metadata)
    engine = create_engine(DATABASE_URL)
    query = f"""
        SELECT 
            f.*,
            fl.genotype,
            fl.sex,
            fl.treatment,
            fl.monitor,
            fl.channel
        FROM features f
        JOIN flies fl ON f.fly_id = fl.fly_id AND f.experiment_id = fl.experiment_id
        WHERE f.experiment_id = {experiment_id}
    """
    ML_features = pd.read_sql(query, engine)
    engine.dispose()
    
    if ML_features is None or len(ML_features) == 0:
        raise ValueError(f"No features found in database for experiment_id {experiment_id}")
    
    # Normalize column names to lowercase (database may have mixed case)
    ML_features.columns = [col.lower() if isinstance(col, str) else col for col in ML_features.columns]
    
    # Verify that metadata columns are present
    required_meta_cols = ['genotype', 'sex', 'treatment']
    missing_meta = [col for col in required_meta_cols if col not in ML_features.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns after loading: {missing_meta}. Available columns: {list(ML_features.columns)}")
    
    # Check for NaN values in metadata
    meta_nan = ML_features[required_meta_cols].isna().sum()
    if meta_nan.sum() > 0:
        print(f"Warning: Found NaN values in metadata columns:\n{meta_nan}")
    
    original_count = len(ML_features)
    
    # ============================================================
    # STEP 2: Remove problematic flies
    # ============================================================
    print(f"\n[Step 2] Removing problematic flies...")
    ML_features, removed_flies = remove_problematic_flies(ML_features)
    print(f"  Removed {len(removed_flies)} problematic flies")
    print(f"  Remaining: {len(ML_features)} flies")
    
    if len(ML_features) == 0:
        raise ValueError("All flies were removed as problematic. Cannot proceed with IQR outlier detection.")
    
    # ============================================================
    # STEP 3: Remove IQR outliers
    # ============================================================
    print(f"\n[Step 3] Removing IQR outliers...")
    before_iqr = len(ML_features)
    ML_features, iqr_bounds, outlier_summary = remove_iqr_outliers(
        ML_features, 
        column='total_sleep_mean',
        multiplier=iqr_multiplier
    )
    after_iqr = len(ML_features)
    print(f"  Removed {before_iqr - after_iqr} IQR outliers")
    print(f"  Remaining: {after_iqr} flies")
    
    # ============================================================
    # STEP 4: Fix NaN values
    # ============================================================
    ML_features = fix_nan_values(ML_features)
    
    # ============================================================
    # STEP 5: Create z-scored features
    # ============================================================
    ML_features_Z = create_z_scored_features(ML_features)
    
    # ============================================================
    # STEP 6: Run diagnostics
    # ============================================================
    diagnostics = run_diagnostics(ML_features)
    
    # ============================================================
    # STEP 7: Save outputs to database
    # ============================================================
    if USE_DATABASE and DB_AVAILABLE and experiment_id:
        try:
            engine = create_engine(DATABASE_URL)
            
            # Update cleaned features in features table using bulk UPSERT
            ML_features_db = ML_features.copy()
            ML_features_db['experiment_id'] = experiment_id
            
            # Metadata columns that should NOT be saved to features table (they're in flies table)
            metadata_cols = ['genotype', 'sex', 'treatment', 'monitor', 'channel']
            
            # Map column names (remove _z suffix if present, keep original names, exclude metadata)
            feature_cols = [col for col in ML_features_db.columns 
                          if col not in ['fly_id', 'experiment_id', 'feature_id'] + metadata_cols
                          and not col.endswith('_z')]
            
            # Prepare data for bulk update (only feature columns, not metadata)
            update_data = ML_features_db[['fly_id', 'experiment_id'] + feature_cols].copy()
            
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    try:
                        # Prepare tuples for bulk update
                        all_cols_list = ['fly_id', 'experiment_id'] + feature_cols
                        # Convert each column to list
                        column_lists = [update_data[col].values.tolist() for col in all_cols_list]
                        # Zip columns together to create tuples
                        update_tuples = list(zip(*column_lists))
                        
                        # Build UPSERT query (INSERT ... ON CONFLICT DO UPDATE)
                        all_cols = ['fly_id', 'experiment_id'] + feature_cols
                        insert_cols = ', '.join(all_cols)
                        placeholders = ', '.join(['%s'] * len(all_cols))
                        update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in feature_cols])
                        
                        upsert_query = f"""
                            INSERT INTO features ({insert_cols})
                            VALUES %s
                            ON CONFLICT (fly_id, experiment_id)
                            DO UPDATE SET {update_set}
                        """
                        
                        # Single bulk UPSERT operation
                        execute_values(
                            cur,
                            upsert_query,
                            update_tuples,
                            template=None,
                            page_size=1000
                        )
                        
                        conn.commit()
                        print(f"  Updated {len(update_data)} features in database")
                        
                    except psycopg2.Error as e:
                        conn.rollback()
                        print(f"  Error updating features: {e}")
                        raise
            
            # Save z-scored features to features_z table
            ML_features_Z_db = ML_features_Z.copy()
            ML_features_Z_db['experiment_id'] = experiment_id
            
            # The DataFrame already has columns with _z suffix, so we just need to ensure they match database schema
            # Database expects lowercase _z, which we now create in create_z_scored_features
            # Database schema only has specific z-scored columns (no rhythmic_days_z, n_days_z, etc.)
            # List of valid z-scored columns in features_z table
            valid_z_cols = [
                # Rhythm / circadian features
                'mesor_mean_z', 'mesor_sd_z', 'amplitude_mean_z', 'amplitude_sd_z',
                'phase_mean_z', 'phase_sd_z',
                'periodogram_period_mean_z', 'periodogram_period_sd_z', 'periodogram_power_mean_z',
                'activity_onset_zt_mean_z', 'activity_onset_zt_sd_z',
                'activity_offset_zt_mean_z', 'activity_offset_zt_sd_z',
                'interdaily_stability_z',
                # Sleep features
                'total_sleep_mean_z', 'day_sleep_mean_z',
                'night_sleep_mean_z', 'total_bouts_mean_z', 'day_bouts_mean_z',
                'night_bouts_mean_z', 'mean_bout_mean_z', 'max_bout_mean_z',
                'mean_day_bout_mean_z', 'max_day_bout_mean_z', 'mean_night_bout_mean_z',
                'max_night_bout_mean_z', 'frag_bouts_per_hour_mean_z',
                'frag_bouts_per_min_sleep_mean_z', 'mean_wake_bout_mean_z',
                'p_wake_mean_z', 'p_doze_mean_z', 'sleep_latency_mean_z', 'waso_mean_z',
            ]
            
            # Select only columns that are in the valid list
            z_feature_cols = [col for col in ML_features_Z_db.columns if col in valid_z_cols]
            
            # Select columns for database (metadata + z-scored features)
            z_db_columns = ['fly_id', 'experiment_id'] + z_feature_cols
            ML_features_Z_db = ML_features_Z_db[z_db_columns]
            
            # Insert or update z-scored features using bulk UPSERT
            if len(ML_features_Z_db) > 0 and len(z_feature_cols) > 0:
                with psycopg2.connect(**DB_CONFIG) as conn:
                    with conn.cursor() as cur:
                        try:
                            # Prepare tuples for bulk insert/update
                            column_lists = [ML_features_Z_db[col].values.tolist() for col in z_db_columns]
                            # Zip columns together to create tuples
                            z_features_tuples = list(zip(*column_lists))
                            
                            # Build UPSERT query (INSERT ... ON CONFLICT DO UPDATE)
                            insert_cols = ', '.join(z_db_columns)
                            placeholders = ', '.join(['%s'] * len(z_db_columns))
                            update_cols_list = [col for col in z_db_columns if col not in ['fly_id', 'experiment_id']]
                            update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols_list])
                            
                            upsert_query = f"""
                                INSERT INTO features_z ({insert_cols})
                                VALUES %s
                                ON CONFLICT (fly_id, experiment_id)
                                DO UPDATE SET {update_set}
                            """
                            
                            # Single bulk UPSERT operation
                            execute_values(
                                cur,
                                upsert_query,
                                z_features_tuples,
                                template=None,
                                page_size=1000
                            )
                            
                            conn.commit()
                            print(f"  Saved {len(ML_features_Z_db)} z-scored features to database")
                            
                        except psycopg2.Error as e:
                            conn.rollback()
                            print(f"  Error saving z-scored features: {e}")
                            raise
            
            engine.dispose()
        except psycopg2.Error as e:
            raise RuntimeError(f"Database error saving cleaned features to database: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving cleaned features to database: {e}")
    
    # Save diagnostics if requested
    if save_diagnostics:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        diag_file = os.path.join(script_dir, f'ML_features_clean_diagnostics_exp{experiment_id}.txt')
        with open(diag_file, 'w') as f:
            f.write("ML FEATURES CLEANING DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Original flies: {original_count}\n")
            f.write(f"Cleaned flies: {len(ML_features)}\n")
            f.write(f"Removed: {original_count - len(ML_features)}\n\n")
            f.write("Removed Flies:\n")
            f.write(removed_flies.to_string(index=False))
            f.write("\n\nIQR Outlier Summary:\n")
            f.write(outlier_summary.to_string(index=False))
    
    return ML_features, ML_features_Z


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 4: Clean ML features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses latest experiment)
  python 4-clean_ml_features.py
  
  # Use specific experiment
  python 4-clean_ml_features.py --experiment-id 1
        """
    )
    
    parser.add_argument('--iqr-multiplier', type=float, default=DEFAULT_IQR_MULTIPLIER,
                       help=f'IQR multiplier for outlier detection (default: {DEFAULT_IQR_MULTIPLIER})')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to use (default: latest experiment)')
    
    args = parser.parse_args()
    
    clean_ml_features(
        iqr_multiplier=args.iqr_multiplier,
        experiment_id=args.experiment_id
    )


if __name__ == '__main__':
    main()


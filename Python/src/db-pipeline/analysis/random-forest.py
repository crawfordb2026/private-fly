#!/usr/bin/env python3
"""
Random Forest Classification Analysis
====================================
Predicts genotype (or other target) from behavioral features using Random Forest.

This script implements:
1. Train/Validation/Test split (80/10/10 or customizable)
2. Random Forest classifier training
3. Hyperparameter tuning on validation set (optional)
4. Performance evaluation on test set
5. Feature importance analysis

Usage:
    python random_forest.py [--experiment-id ID] [--target genotype] [--output-dir analysis_results/random_forest]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid Tkinter issues on headless/CLI runs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
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


def prepare_features_and_target(df, target='genotype', treatment_filter=None):
    """
    Prepare features (X) and target (y) for classification.
    
    Args:
        df: DataFrame with features and metadata
        target: Column to predict ('genotype', 'sex', 'treatment')
        treatment_filter: If specified, filter to this treatment only (e.g., 'VEH')
    
    Returns:
        X: Feature matrix (z-scored features only)
        y: Target vector
        feature_names: List of feature names
    """
    print(f"\n[Prepare] Preparing data for {target} prediction...")
    
    # Filter by treatment if specified
    if treatment_filter:
        df = df[df['treatment'].str.upper() == treatment_filter.upper()].copy()
        print(f"  Filtered to {treatment_filter} treatment: {len(df)} flies")
    
    # Check target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data. Available: {df.columns.tolist()}")
    
    # Get z-scored feature columns
    meta_cols = ['fly_id', 'genotype', 'sex', 'treatment', 'monitor', 'channel', 'experiment_id', 'feature_id']
    feature_cols = [col for col in df.columns if col.endswith('_z') and col not in meta_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No z-scored feature columns found (columns ending with '_z')")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # Remove rows with NaN in features or target
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].copy()
    y = y[mask].copy()
    
    if len(X) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    print(f"  Features: {len(feature_cols)} z-scored features")
    print(f"  Samples: {len(X)} flies")
    print(f"  Target classes: {sorted(y.unique())}")
    print(f"  Class distribution:")
    for cls, count in y.value_counts().items():
        print(f"    {cls}: {count} ({100*count/len(y):.1f}%)")
    
    return X, y, feature_cols


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets.
    
    First splits into train+val / test, then splits train+val into train / val.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set (default: 0.2 = 20%)
        val_size: Proportion of train+val for validation (default: 0.2 = 20% of train+val)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"\n[Split] Splitting data into train/validation/test sets...")
    print(f"  Test size: {test_size*100:.1f}%")
    print(f"  Validation size: {val_size*100:.1f}% of remaining data")
    
    # First split: train+val / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train / val (from train+val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    # Calculate actual proportions
    total = len(X)
    train_pct = len(X_train) / total * 100
    val_pct = len(X_val) / total * 100
    test_pct = len(X_test) / total * 100
    
    print(f"  Final split:")
    print(f"    Train: {len(X_train)} ({train_pct:.1f}%)")
    print(f"    Validation: {len(X_val)} ({val_pct:.1f}%)")
    print(f"    Test: {len(X_test)} ({test_pct:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_random_forest(X_train, y_train, X_val, y_val, tune_hyperparameters=True, random_state=42):
    """
    Train Random Forest classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        tune_hyperparameters: If True, perform grid search
        random_state: Random seed
    
    Returns:
        Trained RandomForestClassifier model
    """
    print(f"\n[Training] Training Random Forest classifier...")
    
    if tune_hyperparameters:
        print("  Performing hyperparameter tuning on validation set...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        val_score = grid_search.score(X_val, y_val)
        print(f"  Validation score: {val_score:.4f}")
        
        model = grid_search.best_estimator_
    else:
        print("  Using default hyperparameters...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        print(f"  Validation score: {val_score:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate model performance on test set and save results.
    
    Args:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save results
    """
    print(f"\n[Evaluation] Evaluating model on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")
    
    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(y_test.unique())
    
    # Save results to file
    results_file = os.path.join(output_dir, 'classification_results.txt')
    with open(results_file, 'w') as f:
        f.write("Random Forest Classification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Set Size: {len(X_test)}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"Classes: {classes}\n")
        f.write(str(cm))
    
    print(f"\n  ✓ Saved results to: {results_file}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={'fontsize': 18}
    )
    plt.title('Confusion Matrix (Test Set)', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"  ✓ Saved confusion matrix: {output_dir}/confusion_matrix.png")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_feature_importance(model, feature_names, output_dir, top_n=28):
    """
    Plot and save feature importance.
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
        output_dir: Directory to save plot
        top_n: Number of top features to show
    """
    print(f"\n[Feature Importance] Extracting top {top_n} features...")
    
    # Get feature importance (avoid multi-threading issues on some Windows setups)
    old_n_jobs = getattr(model, 'n_jobs', None)
    try:
        if old_n_jobs is not None and old_n_jobs != 1:
            model.n_jobs = 1
        importances = model.feature_importances_
    finally:
        if old_n_jobs is not None:
            model.n_jobs = old_n_jobs
    indices = np.argsort(importances)[::-1]
    
    # Create full DataFrame (all features, sorted)
    importance_all_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    # Save all features to CSV
    importance_file = os.path.join(output_dir, 'feature_importance.csv')
    importance_all_df.to_csv(importance_file, index=False)
    print(f"  ✓ Saved feature importance (all features): {importance_file}")
    
    # Plot
    importance_df = importance_all_df.head(top_n).copy()

    # Short, plot-ready labels for y-axis (keep CSV with original names)
    short_labels = {
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
    importance_df['display'] = importance_df['feature'].map(lambda x: short_labels.get(x, x))

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='display', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance (Random Forest)', fontsize=18, fontweight='bold')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    print(f"  ✓ Saved feature importance plot: {output_dir}/feature_importance.png")
    
    return importance_df


def main():
    parser = argparse.ArgumentParser(
        description='Random Forest Classification: Predict genotype/sex/treatment from features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python random_forest.py
  python random_forest.py --experiment-id 1 --target genotype
  python random_forest.py --target sex --treatment-filter VEH --no-tune
        """
    )
    
    parser.add_argument(
        '--experiment-id',
        type=int,
        default=None,
        help='Experiment ID to use (default: latest experiment)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='genotype',
        choices=['genotype', 'sex', 'treatment'],
        help='Target variable to predict (default: genotype)'
    )
    
    parser.add_argument(
        '--treatment-filter',
        type=str,
        default=None,
        help='Filter to specific treatment only (e.g., VEH). If not specified, uses all treatments.'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion for test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Proportion of train+val for validation (default: 0.2)'
    )
    
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (use default parameters)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: analysis_results/random_forest)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / 'analysis_results' / 'random_forest')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RANDOM FOREST CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data_from_db(args.experiment_id)
        
        # Prepare features and target
        X, y, feature_names = prepare_features_and_target(
            df, target=args.target, treatment_filter=args.treatment_filter
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state
        )
        
        # Train model
        model = train_random_forest(
            X_train, y_train, X_val, y_val,
            tune_hyperparameters=not args.no_tune,
            random_state=args.random_state
        )
        
        # Evaluate on test set
        results = evaluate_model(model, X_test, y_test, args.output_dir)
        
        # Feature importance
        importance_df = plot_feature_importance(model, feature_names, args.output_dir, top_n=20)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\n✓ All outputs saved to: {args.output_dir}/")
        print("\nGenerated files:")
        print("  - classification_results.txt")
        print("  - confusion_matrix.png")
        print("  - feature_importance.csv")
        print("  - feature_importance.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
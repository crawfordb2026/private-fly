#!/usr/bin/env python3
"""
Pipeline Step 3: Create Feature Table

This script:
1. Reads cleaned data (from Step 1 or Step 2)
2. Extracts RHYTHM features (circadian):
   - Calculates hourly totals per fly per day per ZT
   - Runs daily cosinor regression and Lomb–Scargle periodogram (per fly per day)
   - Aggregates to per-fly means and SDs
3. Extracts SLEEP features:
   - Calculates daily sleep metrics per fly per day
   - Aggregates to per-fly means
4. Merges rhythm + sleep features into final ML_features table

Output: Saved to database (features table)

This is the final step in the pipeline. The output is ready for ML analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy import stats
from scipy.signal import lombscargle
from sklearn.linear_model import LinearRegression
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

# ============================================================
#   USER CONFIGURATION
# ============================================================


# Feature extraction settings
DEFAULT_EXCLUDE_DAYS = [1, 7]
DEFAULT_SLEEP_THRESHOLD_MIN = 5  # minutes of inactivity defining sleep
DEFAULT_BIN_LENGTH_MIN = 1  # MT resolution in minutes; if DAM files are not 1-min binned, set to bin length
DEFAULT_PERIOD = 24  # Circadian period in hours
DEFAULT_LIGHTS_ON = 9  # Hour when lights turn on (ZT0)


# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def is_ok(x):
    """Check if a value is valid (not NA, not empty, not "na")."""
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na" and x_str != "nan"


# ============================================================
#   RHYTHM FEATURES (CIRCADIAN)
# ============================================================

def prepare_rhythm_data(dam_clean, exclude_days):
    """Prepare data for rhythm analysis (MT only)."""
    df = dam_clean.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
    
    # Normalize column names
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Value': 'value',
        'Genotype': 'genotype', 'Sex': 'sex', 'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Exclude days
    if 'exp_day' in df.columns:
        df = df[~df['exp_day'].isin(exclude_days)].copy()
    
    # Drop rows with missing metadata or zt
    df = df.dropna(subset=['genotype', 'sex', 'treatment', 'zt', 'exp_day'])
    
    # Use fly_id from database if available, otherwise create it (format: M{monitor}_Ch{channel:02d})
    if 'fly_id' not in df.columns or df['fly_id'].isna().all():
        df['fly_id'] = 'M' + df['monitor'].astype(str) + '_Ch' + df['channel'].astype(str).str.zfill(2)
    
    # Convert zt to numeric
    df['zt'] = pd.to_numeric(df['zt'], errors='coerce')
    df = df.dropna(subset=['zt'])
    
    return df


def calculate_hourly_totals(dam_rhythm):
    """Calculate hourly totals per fly per day per ZT."""
    hourly_day = dam_rhythm.groupby(
        ['fly_id', 'genotype', 'sex', 'treatment', 'exp_day', 'zt'],
        as_index=False
    )['value'].sum().rename(columns={'value': 'hourly_MT'})
    
    # Merge back monitor and channel (they're constant per fly_id)
    monitor_channel = dam_rhythm[['fly_id', 'monitor', 'channel']].drop_duplicates()
    hourly_day = hourly_day.merge(monitor_channel, on='fly_id', how='left')
    
    return hourly_day


def run_daily_cosinor(hourly_data, period=24):
    """
    Run cosinor regression for one fly-day.
    
    Model: hourly_MT ~ Mesor + A*cos(2π*ZT/period) + B*sin(2π*ZT/period)
    
    Returns:
        Series with fly_id, exp_day, Mesor, Amp, phase, Cos_p
    """
    df = hourly_data.copy()
    
    # Create cos/sin terms
    df['rad'] = 2 * np.pi * df['zt'] / period
    df['cos_term'] = np.cos(df['rad'])
    df['sin_term'] = np.sin(df['rad'])
    
    # Fit linear regression
    X = df[['cos_term', 'sin_term']].values
    y = df['hourly_MT'].values
    
    if len(y) < 3:
        # Not enough data points
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
            'exp_day': df['exp_day'].iloc[0] if len(df) > 0 else None,
            'monitor': df['monitor'].iloc[0] if len(df) > 0 and 'monitor' in df.columns else None,
            'channel': df['channel'].iloc[0] if len(df) > 0 and 'channel' in df.columns else None,
            'Mesor': np.nan,
            'Amp': np.nan,
            'phase': np.nan,
            'Cos_p': np.nan
        })
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    cos_coef = model.coef_[0]
    sin_coef = model.coef_[1]
    
    # Calculate amplitude
    amplitude = np.sqrt(cos_coef**2 + sin_coef**2)
    
    # Calculate phase (in hours)
    phase_rad = np.arctan2(-sin_coef, cos_coef)
    phase_hours = (period * phase_rad / (2 * np.pi)) % period
    
    # Calculate p-value using F-test
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    n = len(y)
    p = 2  # number of predictors (cos, sin)
    
    if ss_tot > 0 and n > p:
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
        p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    else:
        p_value = np.nan
    
    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'exp_day': df['exp_day'].iloc[0],
        'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
        'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
        'Mesor': intercept,
        'Amp': amplitude,
        'phase': phase_hours,
        'Cos_p': p_value
    })


def run_daily_periodogram(hourly_data):
    """
    Compute Lomb-Scargle periodogram to detect dominant period and rhythm strength.

    Args:
        hourly_data: DataFrame with zt and hourly_MT columns (same format as run_daily_cosinor)

    Returns:
        Series with fly_id, exp_day, monitor, channel, periodogram_period, periodogram_power
        Both periodogram features are np.nan if input contains NaN or insufficient data
    """
    df = hourly_data.copy()

    zt_hours = df['zt'].values
    activity_hourly = df['hourly_MT'].values

    if len(activity_hourly) < 3 or np.isnan(activity_hourly).any() or np.isnan(zt_hours).any():
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
            'exp_day': df['exp_day'].iloc[0] if len(df) > 0 else None,
            'monitor': df['monitor'].iloc[0] if len(df) > 0 and 'monitor' in df.columns else None,
            'channel': df['channel'].iloc[0] if len(df) > 0 and 'channel' in df.columns else None,
            'periodogram_period': np.nan,
            'periodogram_power': np.nan
        })

    y = activity_hourly - np.mean(activity_hourly)

    if np.var(y) == 0 or np.all(y == 0):
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0],
            'exp_day': df['exp_day'].iloc[0],
            'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
            'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
            'periodogram_period': np.nan,
            'periodogram_power': np.nan
        })

    t = zt_hours * 2 * np.pi / 24
    periods = np.linspace(18, 30, 1000)
    freqs = 2 * np.pi / periods
    power = lombscargle(t, y, freqs, normalize=True)
    max_power_idx = np.argmax(power)
    periodogram_period = periods[max_power_idx]
    periodogram_power = power[max_power_idx]

    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'exp_day': df['exp_day'].iloc[0],
        'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
        'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
        'periodogram_period': periodogram_period,
        'periodogram_power': periodogram_power
    })


def run_daily_onset_offset(hourly_data, threshold_sd=1.0, min_bins=2):
    """
    Detect activity onset and offset for one fly-day.

    Onset: first sustained crossing above threshold after the daily minimum.
    Offset: last above-threshold bin followed by a sustained below-threshold period.
    Threshold = 10th-percentile baseline + 1 SD of the smoothed profile.
    """
    df = hourly_data.sort_values('zt').copy()
    zt = df['zt'].values
    y = df['hourly_MT'].values.astype(float)

    null = pd.Series({
        'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
        'exp_day': df['exp_day'].iloc[0] if len(df) > 0 else None,
        'daily_activity_onset_zt': np.nan,
        'daily_activity_offset_zt': np.nan,
    })

    if len(y) < 3 or np.isnan(y).any():
        return null

    smoothed = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    threshold = np.percentile(smoothed, 10) + threshold_sd * np.std(smoothed)
    above = smoothed > threshold

    # Onset: first sustained crossing above threshold after the daily minimum
    onset_zt = np.nan
    min_idx = int(np.argmin(smoothed))
    for i in range(min_idx, len(above) - min_bins + 1):
        if np.all(above[i:i + min_bins]):
            onset_zt = zt[i]
            break

    # Offset: last above-threshold bin followed by a sustained below-threshold period
    offset_zt = np.nan
    for i in range(len(above) - min_bins - 1, -1, -1):
        if above[i] and np.all(~above[i + 1:i + 1 + min_bins]):
            offset_zt = zt[i]
            break

    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'exp_day': df['exp_day'].iloc[0],
        'daily_activity_onset_zt': onset_zt,
        'daily_activity_offset_zt': offset_zt,
    })


def compute_interdaily_stability(hourly_day):
    """
    Compute interdaily stability (IS) per fly from the full multi-day hourly time series.

    IS = [n * sum_h(mean_h - grand_mean)^2] / [p * sum_i(x_i - grand_mean)^2]
    where p = hours per day, n = total hourly bins, mean_h = mean activity at each ZT hour.
    """
    results = []
    for fly_id, group in hourly_day.groupby('fly_id'):
        x = group['hourly_MT'].values.astype(float)

        if len(x) < 2 or np.isnan(x).any():
            results.append({'fly_id': fly_id, 'interdaily_stability': np.nan})
            continue

        grand_mean = np.mean(x)
        denom = np.sum((x - grand_mean) ** 2)

        if denom == 0:
            results.append({'fly_id': fly_id, 'interdaily_stability': np.nan})
            continue

        hourly_means = group.groupby('zt')['hourly_MT'].mean()
        p = len(hourly_means)
        n = len(x)
        is_val = (n * np.sum((hourly_means.values - grand_mean) ** 2)) / (p * denom)
        results.append({'fly_id': fly_id, 'interdaily_stability': is_val})

    return pd.DataFrame(results)


def compute_rhythm_features(dam_clean, exclude_days, period):
    """Compute per-fly rhythm features (daily cosinor + periodogram, then aggregate)."""
    # Prepare data
    dam_rhythm = prepare_rhythm_data(dam_clean, exclude_days)

    # Calculate hourly totals
    hourly_day = calculate_hourly_totals(dam_rhythm)

    # Compute interdaily stability from full cross-day series (one value per fly)
    is_features = compute_interdaily_stability(hourly_day)

    # Run daily cosinor, periodogram, and onset/offset for each fly-day
    daily_cosinor_list = []
    daily_periodogram_list = []
    daily_onset_offset_list = []

    for (fly_id, exp_day), group in hourly_day.groupby(['fly_id', 'exp_day']):
        cosinor_result = run_daily_cosinor(group, period)
        periodogram_result = run_daily_periodogram(group)
        onset_offset_result = run_daily_onset_offset(group)
        daily_cosinor_list.append(cosinor_result)
        daily_periodogram_list.append(periodogram_result)
        daily_onset_offset_list.append(onset_offset_result)

    daily_cosinor = pd.DataFrame(daily_cosinor_list)
    daily_periodogram = pd.DataFrame(daily_periodogram_list)
    daily_onset_offset = pd.DataFrame(daily_onset_offset_list)

    # Get metadata (preserve monitor and channel)
    metadata = dam_rhythm[['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment']].drop_duplicates()

    # Merge all daily results
    daily_features = daily_cosinor.merge(
        daily_periodogram[['fly_id', 'exp_day', 'periodogram_period', 'periodogram_power']],
        on=['fly_id', 'exp_day'],
        how='left'
    ).merge(
        daily_onset_offset[['fly_id', 'exp_day', 'daily_activity_onset_zt', 'daily_activity_offset_zt']],
        on=['fly_id', 'exp_day'], how='left'
    )

    # Merge metadata - ensure monitor/channel are present (from Series or metadata)
    if 'monitor' in daily_features.columns and 'channel' in daily_features.columns:
        daily_features = daily_features.merge(metadata[['fly_id', 'genotype', 'sex', 'treatment']], on='fly_id', how='left')
        if daily_features['monitor'].isna().any() or daily_features['channel'].isna().any():
            monitor_channel = metadata[['fly_id', 'monitor', 'channel']]
            daily_features = daily_features.drop(columns=['monitor', 'channel'], errors='ignore')
            daily_features = daily_features.merge(monitor_channel, on='fly_id', how='left')
    else:
        daily_features = daily_features.merge(metadata, on='fly_id', how='left')

    # Aggregate to per-fly means and SDs
    cosinor_features = daily_features.groupby('fly_id').agg({
        'monitor': 'first',
        'channel': 'first',
        'genotype': 'first',
        'sex': 'first',
        'treatment': 'first',
        'Mesor': ['mean', 'std'],
        'Amp': ['mean', 'std'],
        'phase': ['mean', 'std'],
        'Cos_p': lambda x: (x < 0.05).sum(),  # rhythmic_days
        'periodogram_period': ['mean', 'std'],
        'periodogram_power': 'mean', # rhythmicity
        'daily_activity_onset_zt': ['mean', 'std'],
        'daily_activity_offset_zt': ['mean', 'std'],
    }).reset_index()

    # Flatten column names (all lowercase)
    cosinor_features.columns = [
        'fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment',
        'mesor_mean', 'mesor_sd', 'amplitude_mean', 'amplitude_sd',
        'phase_mean', 'phase_sd', 'rhythmic_days',
        'periodogram_period_mean', 'periodogram_period_sd', 'periodogram_power_mean',
        'activity_onset_zt_mean', 'activity_onset_zt_sd',
        'activity_offset_zt_mean', 'activity_offset_zt_sd',
    ]

    # Merge interdaily stability (one value per fly, computed across all days)
    cosinor_features = cosinor_features.merge(is_features, on='fly_id', how='left')

    return cosinor_features


# ============================================================
#   SLEEP FEATURES
# ============================================================

def prepare_sleep_data(dam_clean, exclude_days):
    """Prepare data for sleep analysis (MT only)."""
    df = dam_clean.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
    
    # Normalize column names
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Value': 'value',
        'Genotype': 'genotype', 'Sex': 'sex', 'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Exclude days
    if 'exp_day' in df.columns:
        df = df[~df['exp_day'].isin(exclude_days)].copy()
    
    # Use fly_id from database if available, otherwise create it (format: M{monitor}_Ch{channel:02d})
    if 'fly_id' not in df.columns or df['fly_id'].isna().all():
        df['fly_id'] = 'M' + df['monitor'].astype(str) + '_Ch' + df['channel'].astype(str).str.zfill(2)
    df['zt_num'] = pd.to_numeric(df['zt'], errors='coerce')
    
    # Rename value to movement
    df = df.rename(columns={'value': 'movement'})
    
    # Sort by fly_id, exp_day, datetime
    df = df.sort_values(['fly_id', 'exp_day', 'datetime']).reset_index(drop=True)
    
    return df


def compute_sleep_features_daily(df, bin_length_min, sleep_threshold_min):
    """
    Compute daily sleep features for one fly-day.
    """
    df = df.copy().reset_index(drop=True)
    
    # Detect inactivity and sleep
    df['inactive'] = (df['movement'] == 0)
    df['run_id'] = (df['inactive'] != df['inactive'].shift(1, fill_value=False)).cumsum()
    df['run_len_min'] = df.groupby('run_id')['inactive'].transform('count') * bin_length_min
    df['sleep'] = df['inactive'] & (df['run_len_min'] >= sleep_threshold_min)
    df['is_day'] = df['zt_num'] < 12
    
    # Detect sleep bout starts
    df['start'] = df['sleep'] & (~df['sleep'].shift(1, fill_value=False))
    df['bout_id'] = (df['start'].cumsum() * df['sleep']).replace(0, np.nan)
    
    # Extract bouts
    # Use as_index=False to keep grouping columns as regular columns (avoids reset_index conflict)
    bouts = df[df['sleep'] & df['bout_id'].notna()].groupby(['fly_id', 'exp_day', 'bout_id'], as_index=False).agg({
        'bout_id': 'count',  # Count bins per bout (bout_len_min will be calculated)
        'is_day': 'first'
    })
    # Rename the aggregated bout_id count to avoid name conflict with grouping column
    bouts = bouts.rename(columns={'bout_id': 'bout_count'})
    bouts['bout_len_min'] = bouts['bout_count'] * bin_length_min
    
    # Calculate metrics
    total_bouts = len(bouts)
    day_bouts = bouts['is_day'].sum()
    night_bouts = (~bouts['is_day']).sum()
    
    total_sleep_min = df['sleep'].sum() * bin_length_min
    day_sleep_min = (df['sleep'] & df['is_day']).sum() * bin_length_min
    night_sleep_min = (df['sleep'] & ~df['is_day']).sum() * bin_length_min
    
    total_hours = len(df) * bin_length_min / 60
    
    mean_bout_min = bouts['bout_len_min'].mean() if total_bouts > 0 else np.nan
    max_bout_min = bouts['bout_len_min'].max() if total_bouts > 0 else np.nan
    
    mean_day_bout_min = bouts[bouts['is_day']]['bout_len_min'].mean() if day_bouts > 0 else np.nan
    max_day_bout_min = bouts[bouts['is_day']]['bout_len_min'].max() if day_bouts > 0 else np.nan
    
    mean_night_bout_min = bouts[~bouts['is_day']]['bout_len_min'].mean() if night_bouts > 0 else np.nan
    max_night_bout_min = bouts[~bouts['is_day']]['bout_len_min'].max() if night_bouts > 0 else np.nan
    
    fragmentation_hour = total_bouts / total_hours if total_hours > 0 else np.nan
    fragmentation_min_sleep = total_bouts / total_sleep_min if total_sleep_min > 0 else np.nan
    
    # Transition probabilities
    sleep_vec = df['sleep'].values
    N = len(sleep_vec)
    N_S = sleep_vec.sum()
    N_W = (~sleep_vec).sum()
    
    if N > 1:
        N_S_to_W = ((~sleep_vec[1:]) & sleep_vec[:-1]).sum()
        N_W_to_S = (sleep_vec[1:] & (~sleep_vec[:-1])).sum()
    else:
        N_S_to_W = 0
        N_W_to_S = 0
    
    P_wake = N_S_to_W / N_S if N_S > 0 else np.nan
    P_doze = N_W_to_S / N_W if N_W > 0 else np.nan
    
    # Sleep latency and WASO (from dark phase)
    dark_df = df[(df['zt_num'] >= 12) & (df['zt_num'] < 24)].reset_index(drop=True)
    
    if len(dark_df) > 0 and dark_df['sleep'].any():
        idx = dark_df['sleep'].idxmax()
        sleep_latency_min = idx * bin_length_min
        WASO_min = (~dark_df.loc[idx:, 'sleep']).sum() * bin_length_min
    else:
        sleep_latency_min = np.nan
        WASO_min = np.nan
    
    # Mean wake bout length
    df['wake'] = ~df['sleep']
    df['wake_run'] = (df['wake'] != df['wake'].shift(1, fill_value=False)).cumsum()
    wake_bouts = df[df['wake']].groupby('wake_run').size() * bin_length_min
    mean_wake_bout_min = wake_bouts.mean() if len(wake_bouts) > 0 else np.nan
    
    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'exp_day': df['exp_day'].iloc[0],
        'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
        'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
        'total_sleep_min': total_sleep_min,
        'day_sleep_min': day_sleep_min,
        'night_sleep_min': night_sleep_min,
        'total_bouts': total_bouts,
        'day_bouts': day_bouts,
        'night_bouts': night_bouts,
        'mean_bout_min': mean_bout_min,
        'max_bout_min': max_bout_min,
        'mean_day_bout_min': mean_day_bout_min,
        'max_day_bout_min': max_day_bout_min,
        'mean_night_bout_min': mean_night_bout_min,
        'max_night_bout_min': max_night_bout_min,
        'fragmentation_bouts_per_hour': fragmentation_hour,
        'fragmentation_bouts_per_min_sleep': fragmentation_min_sleep,
        'P_wake': P_wake,
        'P_doze': P_doze,
        'sleep_latency_min': sleep_latency_min,
        'WASO_min': WASO_min,
        'mean_wake_bout_min': mean_wake_bout_min
    })


def compute_sleep_features(dam_clean, exclude_days, bin_length_min, sleep_threshold_min):
    """Compute per-fly sleep features (daily metrics, then aggregate)."""
    # Prepare data
    mt_data = prepare_sleep_data(dam_clean, exclude_days)
    
    # Compute daily sleep features
    daily_sleep_list = []
    
    for (fly_id, exp_day), group in mt_data.groupby(['fly_id', 'exp_day']):
        result = compute_sleep_features_daily(group, bin_length_min, sleep_threshold_min)
        daily_sleep_list.append(result)
    
    daily_sleep_features = pd.DataFrame(daily_sleep_list)
    
    # Get metadata (preserve monitor and channel)
    metadata = mt_data[['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment']].drop_duplicates()
    
    # Merge metadata - ensure monitor/channel are present (from Series or metadata)
    if 'monitor' in daily_sleep_features.columns and 'channel' in daily_sleep_features.columns:
        # Monitor/channel already present from Series, just merge other metadata
        daily_sleep_features = daily_sleep_features.merge(metadata[['fly_id', 'genotype', 'sex', 'treatment']], on='fly_id', how='left')
        # Fill any missing monitor/channel values from metadata
        if daily_sleep_features['monitor'].isna().any() or daily_sleep_features['channel'].isna().any():
            monitor_channel = metadata[['fly_id', 'monitor', 'channel']]
            daily_sleep_features = daily_sleep_features.drop(columns=['monitor', 'channel'], errors='ignore')
            daily_sleep_features = daily_sleep_features.merge(monitor_channel, on='fly_id', how='left')
    else:
        # Monitor/channel missing, merge full metadata
        daily_sleep_features = daily_sleep_features.merge(metadata, on='fly_id', how='left')
    
    # Aggregate to per-fly means
    sleep_ML_features = daily_sleep_features.groupby('fly_id').agg({
        'monitor': 'first',
        'channel': 'first',
        'genotype': 'first',
        'sex': 'first',
        'treatment': 'first',
        'total_sleep_min': 'mean',
        'day_sleep_min': 'mean',
        'night_sleep_min': 'mean',
        'total_bouts': 'mean',
        'day_bouts': 'mean',
        'night_bouts': 'mean',
        'mean_bout_min': 'mean',
        'max_bout_min': 'mean',
        'mean_day_bout_min': 'mean',
        'max_day_bout_min': 'mean',
        'mean_night_bout_min': 'mean',
        'max_night_bout_min': 'mean',
        'fragmentation_bouts_per_hour': 'mean',
        'fragmentation_bouts_per_min_sleep': 'mean',
        'mean_wake_bout_min': 'mean',
        'P_wake': 'mean',
        'P_doze': 'mean',
        'sleep_latency_min': 'mean',
        'WASO_min': 'mean',
        'exp_day': 'count'  # n_days
    }).reset_index()
    
    # Rename columns (use lowercase to match database schema)
    sleep_ML_features = sleep_ML_features.rename(columns={
        'total_sleep_min': 'total_sleep_mean',
        'day_sleep_min': 'day_sleep_mean',
        'night_sleep_min': 'night_sleep_mean',
        'total_bouts': 'total_bouts_mean',
        'day_bouts': 'day_bouts_mean',
        'night_bouts': 'night_bouts_mean',
        'mean_bout_min': 'mean_bout_mean',
        'max_bout_min': 'max_bout_mean',
        'mean_day_bout_min': 'mean_day_bout_mean',
        'max_day_bout_min': 'max_day_bout_mean',
        'mean_night_bout_min': 'mean_night_bout_mean',
        'max_night_bout_min': 'max_night_bout_mean',
        'fragmentation_bouts_per_hour': 'frag_bouts_per_hour_mean',
        'fragmentation_bouts_per_min_sleep': 'frag_bouts_per_min_sleep_mean',
        'mean_wake_bout_min': 'mean_wake_bout_mean',
        'P_wake': 'p_wake_mean',  # Use lowercase to match database schema
        'P_doze': 'p_doze_mean',  # Use lowercase to match database schema
        'sleep_latency_min': 'sleep_latency_mean',
        'WASO_min': 'waso_mean',
        'exp_day': 'n_days'
    })
    
    return sleep_ML_features


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def create_feature_table(
    exclude_days=DEFAULT_EXCLUDE_DAYS,
    sleep_threshold_min=DEFAULT_SLEEP_THRESHOLD_MIN,
    bin_length_min=DEFAULT_BIN_LENGTH_MIN,
    period=DEFAULT_PERIOD,
    experiment_id=None
):
    """
    Main function to create ML feature table.
    
    Args:
        exclude_days: List of days to exclude
        sleep_threshold_min: Minimum minutes of inactivity for sleep
        bin_length_min: Length of each time bin in minutes
        period: Circadian period in hours
        experiment_id: Experiment ID to use (None = use latest)
        
    Returns:
        ML_features DataFrame
    """
    # Require database
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    
    # ============================================================
    # STEP 1: Load data from database
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
    
    # Load data from database
    dam_clean = step1.load_readings_from_db(experiment_id)
    if dam_clean is None or len(dam_clean) == 0:
        raise ValueError(f"No data found in database for experiment_id {experiment_id}")
    
    # ============================================================
    # STEP 2: Compute rhythm features
    # ============================================================
    cosinor_features = compute_rhythm_features(dam_clean, exclude_days, period)
    
    # ============================================================
    # STEP 3: Compute sleep features
    # ============================================================
    sleep_features = compute_sleep_features(dam_clean, exclude_days, bin_length_min, sleep_threshold_min)
    
    # ============================================================
    # STEP 3: Merge features
    # ============================================================
    ML_features = cosinor_features.merge(
        sleep_features,
        on=['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment'],
        how='inner'
    )
    
    # ============================================================
    # STEP 4: Sort and save output
    # ============================================================
    # Sort by monitor and channel (direct numeric sorting - much more efficient)
    ML_features = ML_features.sort_values(['monitor', 'channel'])
    
    # ============================================================
    # STEP 5: Save features to database
    # ============================================================
    if USE_DATABASE and DB_AVAILABLE and experiment_id:
        try:
            engine = create_engine(DATABASE_URL)
            
            # Add experiment_id to features
            ML_features_db = ML_features.copy()
            ML_features_db['experiment_id'] = experiment_id
            
            # Map column names to database schema (all lowercase)
            feature_mapping = {
                'fly_id': 'fly_id',
                'experiment_id': 'experiment_id',
                'mesor_mean': 'mesor_mean',
                'mesor_sd': 'mesor_sd',
                'amplitude_mean': 'amplitude_mean',
                'amplitude_sd': 'amplitude_sd',
                'phase_mean': 'phase_mean',
                'phase_sd': 'phase_sd',
                'rhythmic_days': 'rhythmic_days',
                'periodogram_period_mean': 'periodogram_period_mean',
                'periodogram_period_sd': 'periodogram_period_sd',
                'periodogram_power_mean': 'periodogram_power_mean',
                'activity_onset_zt_mean': 'activity_onset_zt_mean',
                'activity_onset_zt_sd': 'activity_onset_zt_sd',
                'activity_offset_zt_mean': 'activity_offset_zt_mean',
                'activity_offset_zt_sd': 'activity_offset_zt_sd',
                'interdaily_stability': 'interdaily_stability',
                'total_sleep_mean': 'total_sleep_mean',
                'day_sleep_mean': 'day_sleep_mean',
                'night_sleep_mean': 'night_sleep_mean',
                'total_bouts_mean': 'total_bouts_mean',
                'day_bouts_mean': 'day_bouts_mean',
                'night_bouts_mean': 'night_bouts_mean',
                'mean_bout_mean': 'mean_bout_mean',
                'max_bout_mean': 'max_bout_mean',
                'mean_day_bout_mean': 'mean_day_bout_mean',
                'max_day_bout_mean': 'max_day_bout_mean',
                'mean_night_bout_mean': 'mean_night_bout_mean',
                'max_night_bout_mean': 'max_night_bout_mean',
                'frag_bouts_per_hour_mean': 'frag_bouts_per_hour_mean',
                'frag_bouts_per_min_sleep_mean': 'frag_bouts_per_min_sleep_mean',
                'mean_wake_bout_mean': 'mean_wake_bout_mean',
                'p_wake_mean': 'p_wake_mean',
                'p_doze_mean': 'p_doze_mean',
                'sleep_latency_mean': 'sleep_latency_mean',
                'waso_mean': 'waso_mean'
            }
            
            # Select and rename columns to match database schema
            db_columns = [col for col in feature_mapping.keys() if col in ML_features_db.columns]
            ML_features_db = ML_features_db[db_columns].rename(columns=feature_mapping)
            
            print(f"Saving {len(ML_features_db)} flies (feature rows) to database for experiment_id {experiment_id}")
            print(f"Feature columns: {len([c for c in ML_features_db.columns if c not in ['fly_id', 'experiment_id']])} features per fly")
            print(f"Columns to save: {list(ML_features_db.columns)}")
            
            if len(ML_features_db) == 0:
                print("WARNING: No features to save!")
                return ML_features
            
            # Use bulk UPSERT (INSERT ... ON CONFLICT DO UPDATE) for features
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    try:
                        # Prepare all data as tuples for bulk insert
                        # Get column names in correct order
                        column_names = list(ML_features_db.columns)
                        feature_cols = [col for col in column_names if col not in ['fly_id', 'experiment_id']]
                        
                        # Prepare tuples (all columns in order)
                        # Convert each column to list 
                        column_lists = [ML_features_db[col].values.tolist() for col in column_names]
                        # Zip columns together to create tuples
                        features_tuples = list(zip(*column_lists))
                        
                        # Build UPSERT query (INSERT ... ON CONFLICT DO UPDATE)
                        insert_cols = ', '.join(column_names)
                        placeholders = ', '.join(['%s'] * len(column_names))
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
                            features_tuples,
                            template=None,
                            page_size=1000
                        )
                        
                        conn.commit()
                        
                        # Verify actual count in database (more reliable than rowcount for bulk operations)
                        cur.execute("SELECT COUNT(*) FROM features WHERE experiment_id = %s", (experiment_id,))
                        actual_count = cur.fetchone()[0]
                        print(f"Database save complete: {actual_count} flies saved (inserted or updated)")
                        
                        # Verify all flies were saved
                        if actual_count != len(ML_features_db):
                            print(f"⚠️  WARNING: Expected {len(ML_features_db)} flies, but database has {actual_count} flies")
                        else:
                            print(f"✓ Verified: All {len(ML_features_db)} flies successfully saved to database")
                        
                    except psycopg2.Error as e:
                        conn.rollback()
                        print(f"ERROR saving features: {e}")
                        raise
            
            engine.dispose()
        except psycopg2.Error as e:
            raise RuntimeError(f"Database error saving features to database: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving features to database: {e}")
    
    return ML_features


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 3: Create ML feature table',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--exclude-days', nargs='+', type=int, default=DEFAULT_EXCLUDE_DAYS,
                       help=f'Days to exclude (default: {DEFAULT_EXCLUDE_DAYS})')
    parser.add_argument('--sleep-threshold', type=int, default=DEFAULT_SLEEP_THRESHOLD_MIN,
                       help=f'Minimum minutes of inactivity for sleep (default: {DEFAULT_SLEEP_THRESHOLD_MIN})')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to use (default: latest experiment)')
    
    args = parser.parse_args()
    
    create_feature_table(
        exclude_days=args.exclude_days,
        sleep_threshold_min=args.sleep_threshold,
        experiment_id=args.experiment_id
    )


if __name__ == "__main__":
    main()


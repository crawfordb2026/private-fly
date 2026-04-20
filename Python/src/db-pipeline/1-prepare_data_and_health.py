#!/usr/bin/env python3
"""
Pipeline Step 1: Prepare Data and Generate Health Report

This script:
1. Loads raw DAM files (Monitor5.txt, Monitor6.txt) and metadata (details.txt)
2. Merges data into long format
3. Calculates time variables: Date, Time, ZT (Zeitgeber Time), Phase (Light/Dark)
4. Optionally filters by date range
5. Calculates Exp_Day (experimental day) using global experiment start
6. Generates health report (using in-memory data, no file I/O)

Output: Saved to database (experiments, flies, readings, health_reports tables)

This is the first step in the pipeline. The output is used by:
- Step 2: Fly removal (optional)
- Step 3: Feature extraction
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import csv
from datetime import datetime, date
from pathlib import Path
from io import StringIO

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
#   DATA LOADING FUNCTIONS
# ============================================================

def parse_details(filepath):
    """
    Parse details.txt to extract fly metadata.
    
    Handles space-separated or tab-separated values. Treatment (last column)
    can contain spaces (e.g., "2mM Arg", "2mM His").
    
    Args:
        filepath (str): Path to details.txt file
        
    Returns:
        pd.DataFrame: fly_metadata with columns:
            monitor, channel, fly_id, genotype, sex, treatment
    """
    # Read file and parse manually to handle spaces in treatment field
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # Split on whitespace (handles both tabs and spaces)
        parts = line.split()
        
        if len(parts) < 4:
            continue  # Skip malformed lines
        
        # First 4 parts are: Monitor, Channel, Genotype, Sex
        # Everything after is Treatment (can contain spaces)
        monitor = parts[0]
        channel = parts[1]
        genotype = parts[2]
        sex = parts[3]
        treatment = ' '.join(parts[4:]) if len(parts) > 4 else ''
        
        rows.append({
            'Monitor': monitor,
            'Channel': channel,
            'Genotype': genotype,
            'Sex': sex,
            'Treatment': treatment
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Clean up the data
    # Keep monitor as string to support formats like "51_06_20_25"
    df['monitor'] = df['Monitor'].astype(str)
    df['channel'] = df['Channel'].str.replace('ch', '').astype(int)
    df['genotype'] = df['Genotype']
    df['sex'] = df['Sex']
    df['treatment'] = df['Treatment']
    
    # Create fly_id: M{monitor}_Ch{channel:02d} (vectorized for performance)
    df['fly_id'] = 'M' + df['monitor'].astype(str) + '_Ch' + df['channel'].astype(str).str.zfill(2)
    
    # Select and reorder columns
    fly_metadata = df[['monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment']].copy()
    
    # Remove rows with NA values (empty channels) - vectorized
    # is_ok checks: not NA, not empty, not "na"/"nan"
    genotype_ok = (
        fly_metadata['genotype'].notna() & 
        (fly_metadata['genotype'].astype(str).str.lower() != '') &
        (fly_metadata['genotype'].astype(str).str.lower() != 'na') &
        (fly_metadata['genotype'].astype(str).str.lower() != 'nan')
    )
    fly_metadata = fly_metadata[genotype_ok].copy()
    
    return fly_metadata


def parse_monitor_file(filepath, monitor_num):
    """
    Parse one Monitor*.txt file to extract time-series data in LONG format.
    
    Creates long format data where each timestamp has 3 rows per channel:
    - One row for MT, one for CT, one for Pn
    
    Args:
        filepath (str): Path to Monitor*.txt file
        monitor_num (str): Monitor identifier (e.g., "51_06_20_25" 51 is monitor number rest is data)
        
    Returns:
        pd.DataFrame: time_series_data with columns:
            datetime, monitor, channel, reading, value
    """
    # Read the monitor file
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    # Define column names based on the data structure
    # Columns: ID, date, time, port, [unknowns], movement_type, 0, 0, [32 channel values]
    columns = ['id', 'date', 'time', 'port', 'unknown1', 'unknown2', 'unknown3', 'movement_type', 'zero1', 'zero2']
    # Add 32 channel columns (channels 1-32)
    for i in range(1, 33):
        columns.append(f'ch{i}')
    
    df.columns = columns
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d %b %y %H:%M:%S')
    
    # Filter for the three movement types: MT, CT, Pn
    movement_types = ['MT', 'CT', 'Pn']
    df_filtered = df[df['movement_type'].isin(movement_types)].copy()
    
    # Create timestamp key for grouping
    df_filtered['timestamp_key'] = (
        df_filtered['id'].astype(str) + '_' + 
        df_filtered['date'].astype(str) + '_' + 
        df_filtered['time'].astype(str)
    )
    
    # Stack channels to long format first (vectorized operation)
    channel_cols = [f'ch{i}' for i in range(1, 33)]
    id_vars = ['id', 'date', 'time', 'datetime', 'timestamp_key', 'movement_type']
    
    df_stacked = df_filtered.melt(
        id_vars=id_vars,
        value_vars=channel_cols,
        var_name='channel',
        value_name='value'
    )
    
    # Extract channel number from 'ch1', 'ch2', etc.
    df_stacked['channel'] = df_stacked['channel'].str.replace('ch', '').astype(int)
    
    # Pivot to get MT, CT, Pn as columns for each timestamp and channel
    # This reshapes the data efficiently using vectorized operations
    df_pivot = df_stacked.pivot_table(
        index=['id', 'date', 'time', 'datetime', 'timestamp_key', 'channel'],
        columns='movement_type',
        values='value',
        aggfunc='first',
        fill_value=0
    ).reset_index()
    
    # Flatten MultiIndex columns if present (pivot_table can create MultiIndex)
    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = df_pivot.columns.get_level_values(-1)
    
    # Ensure MT, CT, Pn columns exist (fill with 0 if missing)
    for reading_type in ['MT', 'CT', 'Pn']:
        if reading_type not in df_pivot.columns:
            df_pivot[reading_type] = 0
    
    # Filter: only keep channels where at least one reading type > 0 (active channel)
    mask = (df_pivot['MT'] > 0) | (df_pivot['CT'] > 0) | (df_pivot['Pn'] > 0)
    df_pivot = df_pivot[mask].copy()
    
    # Melt to create one row per reading type (MT, CT, Pn)
    time_series_df = df_pivot.melt(
        id_vars=['datetime', 'channel'],
        value_vars=['MT', 'CT', 'Pn'],
        var_name='reading',
        value_name='value'
    )
    
    # Add monitor and ensure correct types
    time_series_df['monitor'] = monitor_num
    time_series_df['value'] = time_series_df['value'].astype(int)
    
    # Select and reorder columns to match original output format
    time_series_df = time_series_df[['datetime', 'monitor', 'channel', 'reading', 'value']].copy()
    
    return time_series_df


# ============================================================
#   USER CONFIGURATION
# ============================================================

# Default file paths (relative to script location)
def get_default_monitor_files():
    """Get all monitor files from Monitors_date_filtered folder."""
    script_dir = Path(__file__).parent
    monitors_dir = script_dir.parent.parent / 'Monitors_date_filtered'
    monitor_files = sorted(monitors_dir.glob('Monitor*.txt'))
    # Use relative path from script_dir (../../Monitors_date_filtered/filename.txt)
    return [f'../../Monitors_date_filtered/{f.name}' for f in monitor_files] if monitor_files else []

DEFAULT_DAM_FILES = get_default_monitor_files()
DEFAULT_META_PATH = '../../details.txt'
# Database-only pipeline - data saved to database only

# Light cycle settings
DEFAULT_LIGHTS_ON = 9
DEFAULT_LIGHTS_OFF = 21

# Date filtering (optional)
DEFAULT_APPLY_DATE_FILTER = False
DEFAULT_EXP_START = None  # None = auto-detect from data
DEFAULT_EXP_END = None    # None = auto-detect from data

# Health report settings
DEFAULT_BIN_LENGTH_MIN = 1  # If DAM files are not 1-min binned, set to bin length in minutes
DEFAULT_EXCLUDE_DAYS = [1, 7]
DEFAULT_REF_DAY = 4
DEFAULT_DECLINE_THRESHOLD = 0.5
DEFAULT_DEATH_THRESHOLD = 0.2
DEFAULT_TRANSITION_WINDOW = 10  # minutes

# Thresholds for health classification
THRESHOLDS = {
    "A1": 12 * 60 / DEFAULT_BIN_LENGTH_MIN,  # 720 (12 hours in bins)
    "A2": 24 * 60 / DEFAULT_BIN_LENGTH_MIN,  # 1440 (24 hours in bins)
    "ACTIVITY_LOW": 50,
    "INDEX_LOW": 0.02,
    "SLEEP_MAX": 1300,
    "SLEEP_BOUT": 720,
    "MISSING_MAX": 0.10
}


# ============================================================
#   DATA PREPARATION FUNCTIONS
# ============================================================

def calculate_zt_phase(datetime_series, lights_on):
    """
    Calculate ZT (Zeitgeber Time) and Phase (Light/Dark).
    
    ZT is truncated to integer (0-23) based on hour boundaries.
    Each ZT value spans a full hour: 9:00-9:59 → ZT0, 10:00-10:59 → ZT1, etc.
    Phase is "Light" if ZT < 12, "Dark" otherwise.
    
    Args:
        datetime_series: Series of datetime objects
        lights_on: Hour when lights turn on (default 9)
        
    Returns:
        tuple: (ZT, Phase) where ZT is integer 0-23, Phase is "Light" or "Dark"
    """
    hours = datetime_series.dt.hour
    minutes = datetime_series.dt.minute
    hour_local = hours + minutes / 60
    
    # Calculate ZT_raw
    zt_raw = (hour_local - lights_on) % 24
    
    # Truncate to integer (floor), handle 24 -> 0
    zt = np.floor(zt_raw).astype(int)
    zt = np.where(zt == 24, 0, zt)
    
    # Calculate Phase
    phase = np.where(zt_raw < 12, "Light", "Dark")
    
    return zt, phase


def apply_date_filter(df, apply_filter, exp_start, exp_end):
    """
    Apply optional date filtering to the dataset.
    
    Args:
        df: DataFrame with datetime column
        apply_filter: Boolean, whether to apply date filter
        exp_start: Start date (date object or None)
        exp_end: End date (date object or None)
        
    Returns:
        tuple: (filtered_df, actual_exp_start, actual_exp_end)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Auto-detect date range if not specified
    all_dates = sorted(df['date'].unique())
    auto_start = min(all_dates) if all_dates else None
    auto_end = max(all_dates) if all_dates else None
    
    if exp_start is None:
        actual_exp_start = auto_start
    else:
        actual_exp_start = exp_start if isinstance(exp_start, date) else pd.to_datetime(exp_start).date()
    
    if exp_end is None:
        actual_exp_end = auto_end
    else:
        actual_exp_end = exp_end if isinstance(exp_end, date) else pd.to_datetime(exp_end).date()
    
    if apply_filter:
        df = df[(df['date'] >= actual_exp_start) & (df['date'] <= actual_exp_end)].copy()
    
    return df, actual_exp_start, actual_exp_end


def calculate_exp_day_global(df, exp_start):
    """
    Calculate exp_day using global experiment start date 
    
    Args:
        df: DataFrame with date column
        exp_start: Experiment start date (date object)
        
    Returns:
        Series with exp_day values
    """
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Calculate days since start
    exp_start_pd = pd.to_datetime(exp_start)
    df['exp_day'] = (pd.to_datetime(df['date']) - exp_start_pd).dt.days + 1
    
    return df['exp_day']


# ============================================================
#   HEALTH REPORT FUNCTIONS
# ============================================================

def rle(seq):
    """Run-length encoding."""
    if len(seq) == 0:
        return np.array([]), np.array([])
    
    # Convert to numpy array to avoid pandas indexing issues
    if isinstance(seq, pd.Series):
        seq_array = seq.values
    else:
        seq_array = np.asarray(seq)
    
    changes = np.diff(seq_array) != 0
    change_indices = np.where(changes)[0] + 1
    indices = np.concatenate(([0], change_indices, [len(seq_array)]))
    lengths = np.diff(indices)
    values = seq_array[indices[:-1]]
    
    return values, lengths


def longest_zero_run(counts, bin_length_min):
    """Calculate longest consecutive zero-activity period."""
    if len(counts) == 0:
        return 0
    
    has_activity = (counts > 0).astype(int)
    values, lengths = rle(has_activity)
    
    zero_runs = lengths[values == 0]
    if len(zero_runs) == 0:
        return 0
    
    return max(zero_runs) * bin_length_min


def total_sleep_minutes(counts, bin_length_min):
    """Calculate total sleep minutes (5+ minute inactivity periods)."""
    if len(counts) == 0:
        return 0
    
    has_activity = (counts > 0).astype(int)
    values, lengths = rle(has_activity)
    
    sleep_runs = lengths[(values == 0) & (lengths >= 5)]
    return sum(sleep_runs) * bin_length_min


def is_ok(x):
    """Check if a value is valid (not NA, not empty, not "na")."""
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na" and x_str != "nan"


def prep_data_for_health(df, exclude_days, bin_length_min):
    """Prepare and filter data for health report analysis."""
    df = df.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
    
    # Ensure datetime is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Exclude specified days
    if 'exp_day' in df.columns:
        df = df[~df['exp_day'].isin(exclude_days)].copy()
    
    # Clean channel names
    if df['channel'].dtype == 'object':
        df['channel'] = df['channel'].str.replace('^ch', '', regex=True)
    df['channel'] = df['channel'].astype(int)
    
    # Rename value to COUNTS
    df = df.rename(columns={'value': 'COUNTS'})
    
    # Ensure required lowercase columns exist
    required_cols = ['monitor', 'channel', 'genotype', 'sex', 'treatment']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available columns: {list(df.columns)}")
    
    return df


def calculate_daily_metrics(dam_activity, bin_length_min):
    """Calculate daily metrics per fly per day."""
    dam_activity = dam_activity.copy()
    dam_activity['date'] = dam_activity['datetime'].dt.date
    
    # Required columns for grouping (all lowercase)
    required_group_cols = ['monitor', 'channel', 'date']
    optional_group_cols = ['exp_day', 'genotype', 'sex', 'treatment']
    group_cols = required_group_cols + [col for col in optional_group_cols if col in dam_activity.columns]
    
    # Check if COUNTS column exists
    if 'COUNTS' not in dam_activity.columns:
        raise ValueError(f"COUNTS column not found in dam_activity. Available columns: {list(dam_activity.columns)}")
    
    def calc_metrics(group):
        return pd.Series({
            'TOTAL_ACTIVITY': group['COUNTS'].sum(skipna=True),
            'ACTIVITY_INDEX': (group['COUNTS'] > 0).mean(),
            'LONGEST_ZERO': longest_zero_run(group['COUNTS'], bin_length_min),
            'TOTAL_SLEEP': total_sleep_minutes(group['COUNTS'], bin_length_min),
            'MISSING_FRAC': group['COUNTS'].isna().mean()
        })
    
    # Use groupby with include_groups=False to prevent duplicate columns
    grouped = dam_activity.groupby(group_cols, dropna=False)
    daily_summary = grouped.apply(calc_metrics, include_groups=False).reset_index()
    
    # Ensure grouping columns are preserved
    for col in group_cols:
        if col not in daily_summary.columns:
            raise ValueError(f"Grouping column '{col}' missing from daily_summary result. This should not happen.")
    
    # Check if daily_summary is empty or missing columns
    if len(daily_summary) == 0:
        raise ValueError(f"No data in daily_summary. dam_activity has {len(dam_activity)} rows. Group cols: {group_cols}")
    if 'TOTAL_ACTIVITY' not in daily_summary.columns:
        raise ValueError(f"TOTAL_ACTIVITY column missing. Available columns: {list(daily_summary.columns)}")
    
    return daily_summary


def normalize_to_ref_day(daily_summary, ref_day, decline_threshold, death_threshold):
    """Normalize activity to reference day."""
    daily_summary = daily_summary.copy()
    
    ref_activity = daily_summary[daily_summary['exp_day'] == ref_day][
        ['monitor', 'channel', 'TOTAL_ACTIVITY']
    ].rename(columns={'TOTAL_ACTIVITY': 'REF_ACTIVITY'})
    
    daily_summary = daily_summary.merge(ref_activity, on=['monitor', 'channel'], how='left')
    daily_summary['REL_ACTIVITY'] = daily_summary['TOTAL_ACTIVITY'] / daily_summary['REF_ACTIVITY']
    
    def classify_decline(row):
        if pd.isna(row['REF_ACTIVITY']):
            return "No Reference"
        if row['REL_ACTIVITY'] < death_threshold:
            return "Dead (by decline)"
        if row['REL_ACTIVITY'] < decline_threshold:
            return "Unhealthy (by decline)"
        return "Stable"
    
    daily_summary['DECLINE_STATUS'] = daily_summary.apply(classify_decline, axis=1)
    return daily_summary


def startle_test(dam_activity, lights_on, lights_off, transition_window):
    """Test for startle response at light transitions."""
    dam_activity = dam_activity.copy()
    dam_activity['HOUR'] = dam_activity['datetime'].dt.hour + dam_activity['datetime'].dt.minute / 60
    dam_activity['IS_TRANSITION'] = (
        (np.abs(dam_activity['HOUR'] - lights_on) <= transition_window / 60) |
        (np.abs(dam_activity['HOUR'] - lights_off) <= transition_window / 60)
    )
    dam_activity['date'] = dam_activity['datetime'].dt.date
    
    def calc_transition_counts(group):
        transition_rows = group[group['IS_TRANSITION']]
        return pd.Series({
            'TRANSITION_COUNTS': transition_rows['COUNTS'].sum(skipna=True)
        })
    
    transition_data = dam_activity.groupby(['monitor', 'channel', 'date'], group_keys=False).apply(
        calc_transition_counts, include_groups=False
    ).reset_index()
    
    transition_data['NO_STARTLE'] = transition_data['TRANSITION_COUNTS'] == 0
    return transition_data


def classify_status(daily_summary, transition_data, thresholds):
    """Classify fly status using decision tree."""
    daily_summary = daily_summary.copy()
    transition_data = transition_data.copy()
    
    # Ensure required columns exist
    if 'monitor' not in daily_summary.columns or 'channel' not in daily_summary.columns:
        raise ValueError(f"Required columns 'monitor' and 'channel' missing from daily_summary. Available: {list(daily_summary.columns)}")
    if 'monitor' not in transition_data.columns or 'channel' not in transition_data.columns:
        raise ValueError(f"Required columns 'monitor' and 'channel' missing from transition_data. Available: {list(transition_data.columns)}")
    
    fly_status = daily_summary.merge(
        transition_data, on=['monitor', 'channel', 'date'], how='left'
    )
    
    fly_status['NO_STARTLE'] = fly_status['NO_STARTLE'].fillna(False)
    
    fly_status['FLAG_A2'] = fly_status['LONGEST_ZERO'] >= thresholds['A2']
    fly_status['FLAG_A1'] = fly_status['LONGEST_ZERO'] >= thresholds['A1']
    fly_status['FLAG_LOW_ACTIVITY'] = (
        (fly_status['TOTAL_ACTIVITY'] <= thresholds['ACTIVITY_LOW']) |
        (fly_status['ACTIVITY_INDEX'] <= thresholds['INDEX_LOW'])
    )
    fly_status['FLAG_SLEEP'] = (
        (fly_status['TOTAL_SLEEP'] >= thresholds['SLEEP_MAX']) |
        (fly_status['LONGEST_ZERO'] >= thresholds['SLEEP_BOUT'])
    )
    fly_status['FLAG_NO_STARTLE'] = fly_status['NO_STARTLE']
    fly_status['FLAG_MISSING'] = fly_status['MISSING_FRAC'] > thresholds['MISSING_MAX']
    
    # Create fly_id column (vectorized) for tracking unique deaths
    fly_status['_fly_id'] = fly_status['monitor'].astype(str) + '_' + fly_status['channel'].astype(str)
    
    # Initialize STATUS column with default value
    fly_status['STATUS'] = 'Alive'
    
    # Track unique flies that died by each rule (using sets)
    unique_deaths = {'A2': set(), 'A1': set(), 'decline': set()}
    # Also track fly-days for reference
    death_fly_days = {'A2': 0, 'A1': 0, 'decline': 0}
    
    # Step 1: A2 (24+ hours consecutive zero activity) → Dead
    mask_a2 = fly_status['FLAG_A2']
    fly_status.loc[mask_a2, 'STATUS'] = 'Dead'
    unique_deaths['A2'].update(fly_status.loc[mask_a2, '_fly_id'].unique())
    death_fly_days['A2'] = mask_a2.sum()
    
    # Step 2: A1 (12+ hours zero activity AND no startle response) → Dead
    # Only apply if not already marked Dead by A2
    mask_a1 = fly_status['FLAG_A1'] & fly_status['FLAG_NO_STARTLE'] & ~mask_a2
    fly_status.loc[mask_a1, 'STATUS'] = 'Dead'
    unique_deaths['A1'].update(fly_status.loc[mask_a1, '_fly_id'].unique())
    death_fly_days['A1'] = mask_a1.sum()
    
    # Step 3: Check if decline WOULD match (but don't mark as Dead)
    mask_decline = fly_status['DECLINE_STATUS'] == "Dead (by decline)"
    unique_deaths['decline'].update(fly_status.loc[mask_decline, '_fly_id'].unique())
    death_fly_days['decline'] = mask_decline.sum()
    # NOT marking as Dead here - the rule is commented out!
    
    # Step 4: Low activity or excessive sleep AND no startle response → Unhealthy
    # Only apply if still Alive
    mask_unhealthy = (
        (fly_status['FLAG_LOW_ACTIVITY'] | fly_status['FLAG_SLEEP']) & 
        fly_status['FLAG_NO_STARTLE'] & 
        (fly_status['STATUS'] == 'Alive')
    )
    fly_status.loc[mask_unhealthy, 'STATUS'] = 'Unhealthy'
    
    # Step 5: COMMENTED OUT
    # if row['DECLINE_STATUS'] == "Unhealthy (by decline)":
    #     return "Unhealthy"
    
    # Step 6: Too much missing data (> 10%) → QC_Fail
    # Only apply if still Alive
    mask_qc = fly_status['FLAG_MISSING'] & (fly_status['STATUS'] == 'Alive')
    fly_status.loc[mask_qc, 'STATUS'] = 'QC_Fail'
    
    # Step 7: Otherwise, fly is alive (already set as default)
    
    # Clean up temporary column
    fly_status = fly_status.drop(columns=['_fly_id'])
    
    # Calculate total unique flies that died
    all_dead_flies = unique_deaths['A2'] | unique_deaths['A1']
    
    # Print summary
    print(f"\n  Death Classification Summary:")
    print(f"    A2 (24+ hrs zero) → Dead: {len(unique_deaths['A2'])} unique flies")
    print(f"    A1 (12+ hrs zero + no startle) → Dead: {len(unique_deaths['A1'])} unique flies")
    print(f"    Total unique flies marked Dead: {len(all_dead_flies)}")
    
    return fly_status


def apply_irreversible_death(fly_status):
    """Apply irreversible death rule: once Dead, always Dead."""
    fly_status = fly_status.copy()
    
    # Ensure required columns exist
    if 'monitor' not in fly_status.columns or 'channel' not in fly_status.columns:
        raise ValueError(f"Required monitor/channel columns missing. Available: {list(fly_status.columns)}")
    
    fly_status = fly_status.sort_values(['monitor', 'channel', 'exp_day']).reset_index(drop=True)
    
    # Store monitor and channel values before groupby (to restore after)
    monitor_vals = fly_status['monitor'].values
    channel_vals = fly_status['channel'].values
    
    def mark_permanent_death(group):
        group = group.copy()
        dead_found = False
        new_status = []
        for status in group['STATUS']:
            if dead_found:
                new_status.append("Dead")
            elif status == "Dead":
                dead_found = True
                new_status.append("Dead")
            else:
                new_status.append(status)
        group['STATUS'] = new_status
        return group
    
    result = fly_status.groupby(['monitor', 'channel'], group_keys=False).apply(mark_permanent_death, include_groups=False)
    result = result.reset_index(drop=True)
    
    # Restore monitor and channel columns (they're excluded with include_groups=False)
    result['monitor'] = monitor_vals[:len(result)]
    result['channel'] = channel_vals[:len(result)]
    
    return result


def generate_summary(fly_status):
    """Generate per-fly summary table."""
    # Ensure required columns exist (all lowercase)
    required_cols = ['monitor', 'channel', 'genotype', 'sex', 'treatment']
    for col in required_cols:
        if col not in fly_status.columns:
            raise ValueError(f"Required column '{col}' not found in fly_status. Available columns: {list(fly_status.columns)}")
    
    # Vectorized filtering (replaces is_ok.apply() for performance)
    # is_ok checks: not NA, not empty, not "na"/"nan"
    def vectorized_is_ok(series):
        """Vectorized version of is_ok function."""
        series_str = series.astype(str).str.lower()
        return (
            series.notna() & 
            (series_str != '') &
            (series_str != 'na') &
            (series_str != 'nan')
        )
    
    mask = (
        vectorized_is_ok(fly_status['genotype']) &
        vectorized_is_ok(fly_status['sex']) &
        vectorized_is_ok(fly_status['treatment'])
    )
    fly_status_clean = fly_status[mask].copy()
    
    def agg_func(group):
        return pd.Series({
            'DAYS_ANALYZED': len(group),
            'DAYS_ALIVE': (group['STATUS'] == 'Alive').sum(),
            'DAYS_UNHEALTHY': (group['STATUS'] == 'Unhealthy').sum(),
            'DAYS_DEAD': (group['STATUS'] == 'Dead').sum(),
            'DAYS_QC_FAIL': (group['STATUS'] == 'QC_Fail').sum(),
            'FIRST_UNHEALTHY_DAY': group.loc[group['STATUS'] == 'Unhealthy', 'exp_day'].min() if (group['STATUS'] == 'Unhealthy').any() else np.nan,
            'FIRST_DEAD_DAY': group.loc[group['STATUS'] == 'Dead', 'exp_day'].min() if (group['STATUS'] == 'Dead').any() else np.nan,
            'LAST_ALIVE_DAY': group.loc[group['STATUS'] == 'Alive', 'exp_day'].max() if (group['STATUS'] == 'Alive').any() else np.nan,
            'FINAL_STATUS': group['STATUS'].iloc[-1]
        })
    
    health_report = fly_status_clean.groupby(
        ['monitor', 'channel', 'genotype', 'sex', 'treatment'], dropna=False
    ).apply(agg_func, include_groups=False).reset_index()
    
    health_report = health_report.sort_values(['monitor', 'channel']).reset_index(drop=True)
    return health_report


# ============================================================
#   DATABASE FUNCTIONS
# ============================================================

def fly_ids_without_usable_mt_or_pn(dam_merged):
    """
    Fly IDs that should not be written to the database: missing MT or Pn entirely,
    or no non-zero MT, or no non-zero Pn anywhere in the loaded window.

    This drops empty/dead channels and purely flat zero-MT traces (including
    MT=0 with constant non-zero Pn, which still fails the MT rule).
    """
    if dam_merged is None or len(dam_merged) == 0:
        return set()

    if 'fly_id' not in dam_merged.columns or 'reading' not in dam_merged.columns:
        return set()

    all_ids = dam_merged['fly_id'].dropna().astype(str).unique()
    excluded = set()

    mt = dam_merged[dam_merged['reading'] == 'MT']
    pn = dam_merged[dam_merged['reading'] == 'Pn']

    mt_max = mt.groupby('fly_id', sort=False)['value'].max() if len(mt) else pd.Series(dtype=float)
    pn_max = pn.groupby('fly_id', sort=False)['value'].max() if len(pn) else pd.Series(dtype=float)
    mt_max.index = mt_max.index.astype(str)
    pn_max.index = pn_max.index.astype(str)
    mt_ids = set(mt_max.index)
    pn_ids = set(pn_max.index)

    for fid in all_ids:
        fid = str(fid)
        if fid not in mt_ids:
            excluded.add(fid)
            continue
        if fid not in pn_ids:
            excluded.add(fid)
            continue
        m_mt = float(mt_max.loc[fid])
        m_pn = float(pn_max.loc[fid])
        if not np.isfinite(m_mt) or m_mt <= 0:
            excluded.add(fid)
            continue
        if not np.isfinite(m_pn) or m_pn <= 0:
            excluded.add(fid)
            continue

    return excluded


def _fly_id_col_from_monitor_channel(monitor_series, channel_series):
    """Match fly_id convention in parse_details / dam_merged."""
    return (
        'M'
        + monitor_series.astype(str)
        + '_Ch'
        + channel_series.astype(str).str.zfill(2)
    )


def create_experiment(name, start_date, end_date=None, lights_on=9, lights_off=21):
    """Create a new experiment and return experiment_id."""
    if not USE_DATABASE or not DB_AVAILABLE:
        return None
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments (name, start_date, end_date, lights_on_hour, lights_off_hour)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING experiment_id
                """, (name, start_date, end_date, lights_on, lights_off))
                
                experiment_id = cur.fetchone()[0]
                conn.commit()
        
        return experiment_id
    except psycopg2.Error as e:
        print(f"Database error creating experiment: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating experiment: {e}")
        return None

def save_to_database(dam_merged, health_report, fly_status, experiment_id, actual_exp_start):
    """Save data to database."""
    if not USE_DATABASE or not DB_AVAILABLE or experiment_id is None:
        return
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Save flies metadata first (required for foreign key constraint)
        print(f"  Preparing fly metadata...", end='', flush=True)
        flies = dam_merged[['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment']].drop_duplicates()
        flies['experiment_id'] = experiment_id
        print(f" ✓ ({len(flies)} unique flies)")
        
        # Save flies using bulk insert with ON CONFLICT to handle duplicates properly
        # PRIMARY KEY is (fly_id, experiment_id), so use ON CONFLICT (fly_id, experiment_id) DO NOTHING
        print(f"  Inserting flies into database...", end='', flush=True)
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                try:
                    # Prepare all data as tuples in memory for bulk insert
                    flies_tuples = list(zip(
                        flies['fly_id'].astype(str).values.tolist(),
                        flies['experiment_id'].astype(int).values.tolist(),
                        flies['monitor'].astype(str).values.tolist(),  # Keep as string for formats like "51_06_20_25"
                        flies['channel'].astype(int).values.tolist(),
                        flies['genotype'].astype(str).values.tolist(),
                        flies['sex'].astype(str).values.tolist(),
                        flies['treatment'].astype(str).values.tolist()
                    ))
                    
                    # Get count before insert
                    cur.execute("SELECT COUNT(*) FROM flies WHERE experiment_id = %s", (experiment_id,))
                    count_before = cur.fetchone()[0]
                    
                    # Single bulk insert operation (much faster than row-by-row)
                    execute_values(
                        cur,
                        """INSERT INTO flies (fly_id, experiment_id, monitor, channel, genotype, sex, treatment)
                           VALUES %s
                           ON CONFLICT (fly_id, experiment_id) DO NOTHING""",
                        flies_tuples,
                        template=None,
                        page_size=1000  # Process in pages of 1000 for very large datasets
                    )
                    
                    # Get count after insert to determine actual rows inserted
                    cur.execute("SELECT COUNT(*) FROM flies WHERE experiment_id = %s", (experiment_id,))
                    count_after = cur.fetchone()[0]
                    
                    # Count inserted vs skipped
                    total_rows = len(flies_tuples)
                    inserted_count = count_after - count_before  # Actual rows inserted
                    skipped_count = total_rows - inserted_count
                    
                    conn.commit()
                    print(f" ✓ ({inserted_count} new, {skipped_count} duplicates)")
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    print(f"  Error inserting flies: {e}")
                    raise
        
        # Save readings using COPY FROM for bulk insert (much faster than to_sql)
        print(f"  Preparing readings data ({len(dam_merged):,} rows)...", end='', flush=True)
        readings = dam_merged[['datetime', 'fly_id', 'reading', 'value', 'monitor']].copy()
        readings = readings.rename(columns={'reading': 'reading_type'})
        readings['experiment_id'] = experiment_id
        readings = readings[['experiment_id', 'fly_id', 'datetime', 'reading_type', 'value', 'monitor']]
        print(f" ✓")
        
        # Use COPY FROM for bulk insert (10-100x faster than to_sql)
        print(f"  Inserting {len(readings):,} readings into database (this may take several minutes)...", end='', flush=True)
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                try:
                    # Convert DataFrame to CSV-like string in memory
                    output = StringIO()
                    
                    # Ensure datetime is properly formatted for PostgreSQL
                    readings_copy = readings.copy()
                    readings_copy['datetime'] = pd.to_datetime(readings_copy['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Write data to StringIO (no file I/O, all in memory)
                    readings_copy.to_csv(
                        output,
                        sep='\t',           # Tab-separated (PostgreSQL COPY default)
                        header=False,       # No header row
                        index=False,        # No index column
                        quoting=csv.QUOTE_NONE,
                        escapechar='\\',
                        na_rep='\\N',       # PostgreSQL NULL representation
                        doublequote=False
                    )
                    
                    output.seek(0)  # Reset to beginning for reading
                    
                    # Use PostgreSQL COPY FROM (fastest bulk insert method)
                    cur.copy_from(
                        output,
                        'readings',
                        columns=['experiment_id', 'fly_id', 'datetime', 'reading_type', 'value', 'monitor'],
                        sep='\t',
                        null='\\N'
                    )
                    
                    conn.commit()
                    print(f" ✓")
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    print(f"  Error copying readings: {e}")
                    raise
        
        # Save health report (if available) using bulk insert
        if health_report is not None and len(health_report) > 0:
            print(f"  Preparing health report...", end='', flush=True)
            health_db = health_report.copy()
            
            # Create fly_id if not present (vectorized for performance)
            if 'fly_id' not in health_db.columns:
                health_db['fly_id'] = 'M' + health_db['monitor'].astype(str) + '_Ch' + health_db['channel'].astype(str).str.zfill(2)
            
            # Add experiment_id and map columns
            health_db['experiment_id'] = experiment_id
            health_db['report_date'] = actual_exp_start
            health_db['status'] = health_db['FINAL_STATUS']
            
            # To get the detailed metrics, we need to aggregate from fly_status (daily data)
            # Calculate final day metrics for each fly
            if fly_status is not None:
                # Get the last day's metrics for each fly
                fly_last_day = fly_status.groupby(['monitor', 'channel']).last().reset_index()
                
                # Create fly_id in fly_last_day for merging (vectorized for performance)
                fly_last_day['fly_id'] = 'M' + fly_last_day['monitor'].astype(str) + '_Ch' + fly_last_day['channel'].astype(str).str.zfill(2)
                
                # Merge in the detailed metrics
                health_db = health_db.merge(
                    fly_last_day[['fly_id', 'TOTAL_ACTIVITY', 'LONGEST_ZERO', 'REL_ACTIVITY', 
                                  'NO_STARTLE', 'MISSING_FRAC']],
                    on='fly_id',
                    how='left'
                )
                
                # Map to database column names
                health_db['total_activity'] = health_db['TOTAL_ACTIVITY'] if 'TOTAL_ACTIVITY' in health_db.columns else 0
                health_db['longest_zero_hours'] = (health_db['LONGEST_ZERO'] / 60.0) if 'LONGEST_ZERO' in health_db.columns else 0.0  # Convert minutes to hours
                health_db['rel_activity'] = health_db['REL_ACTIVITY'] if 'REL_ACTIVITY' in health_db.columns else np.nan
                health_db['has_startle_response'] = ~health_db['NO_STARTLE'] if 'NO_STARTLE' in health_db.columns else False  # Invert NO_STARTLE
                health_db['missing_fraction'] = health_db['MISSING_FRAC'] if 'MISSING_FRAC' in health_db.columns else np.nan
                
                # Replace inf/-inf with NaN first, then cap values
                # DECIMAL(5,3) can only hold values from -99.999 to 99.999
                # DECIMAL(5,2) can only hold values from -999.99 to 999.99
                for col in ['rel_activity', 'missing_fraction', 'longest_zero_hours']:
                    if col in health_db.columns:
                        # Replace infinity with NaN
                        health_db[col] = health_db[col].replace([np.inf, -np.inf], np.nan)
                        # Cap extreme values to fit in database DECIMAL types
                        if col == 'longest_zero_hours':
                            # DECIMAL(5,2): -999.99 to 999.99
                            health_db[col] = health_db[col].clip(lower=0, upper=999.99)
                        else:
                            # DECIMAL(5,3): -99.999 to 99.999
                            health_db[col] = health_db[col].clip(lower=-99.999, upper=99.999)
            
            print(f" ✓")
            
            # Map additional fields if available - now include all the metric columns
            health_cols = ['fly_id', 'experiment_id', 'report_date', 'status', 
                          'total_activity', 'longest_zero_hours', 'rel_activity',
                          'has_startle_response', 'missing_fraction']
            health_df = health_db[health_cols].copy()
            
            # Ensure report_date is a date object (PostgreSQL expects date type)
            if len(health_df) > 0:
                # Check if dates need conversion (if not already date objects)
                sample_date = health_df['report_date'].iloc[0]
                if not isinstance(sample_date, date):
                    health_df['report_date'] = pd.to_datetime(health_df['report_date']).dt.date
            
            # Use bulk insert with ON CONFLICT to handle duplicates
            print(f"  Inserting health reports...", end='', flush=True)
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    try:
                        # Prepare all data as tuples for bulk insert
                        # Handle None/NaN values properly
                        health_tuples = list(zip(
                            health_df['fly_id'].astype(str).values.tolist(),
                            health_df['experiment_id'].astype(int).values.tolist(),
                            health_df['report_date'].values.tolist(),  # Already date objects, just convert to list
                            health_df['status'].astype(str).values.tolist(),
                            # Convert to Python native types and handle NaN
                            [None if pd.isna(x) else int(x) for x in health_df['total_activity'].values],
                            [None if pd.isna(x) else float(x) for x in health_df['longest_zero_hours'].values],
                            [None if pd.isna(x) else float(x) for x in health_df['rel_activity'].values],
                            [None if pd.isna(x) else bool(x) for x in health_df['has_startle_response'].values],
                            [None if pd.isna(x) else float(x) for x in health_df['missing_fraction'].values]
                        ))
                        
                        # Get count before insert
                        cur.execute("SELECT COUNT(*) FROM health_reports WHERE experiment_id = %s", (experiment_id,))
                        count_before = cur.fetchone()[0]
                        
                        # Single bulk insert operation with all columns
                        execute_values(
                            cur,
                            """INSERT INTO health_reports 
                               (fly_id, experiment_id, report_date, status, 
                                total_activity, longest_zero_hours, rel_activity, 
                                has_startle_response, missing_fraction)
                               VALUES %s
                               ON CONFLICT (fly_id, experiment_id, report_date) 
                               DO UPDATE SET
                                   status = EXCLUDED.status,
                                   total_activity = EXCLUDED.total_activity,
                                   longest_zero_hours = EXCLUDED.longest_zero_hours,
                                   rel_activity = EXCLUDED.rel_activity,
                                   has_startle_response = EXCLUDED.has_startle_response,
                                   missing_fraction = EXCLUDED.missing_fraction""",
                            health_tuples,
                            template=None,
                            page_size=1000
                        )
                        
                        # Get count after insert to determine actual rows inserted
                        cur.execute("SELECT COUNT(*) FROM health_reports WHERE experiment_id = %s", (experiment_id,))
                        count_after = cur.fetchone()[0]
                        
                        inserted_count = count_after - count_before
                        conn.commit()
                        print(f" ✓ ({inserted_count} reports)")
                        
                    except psycopg2.Error as e:
                        conn.rollback()
                        print(f"  Error inserting health reports: {e}")
                        raise
        
        engine.dispose()
        print(f"\n✅ Successfully saved all data to database (experiment_id: {experiment_id})")
    except psycopg2.Error as e:
        # Log database error but continue (allows pipeline to complete even if DB fails)
        print(f"✗ WARNING: Database error saving to database: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        # Log unexpected error but continue
        print(f"✗ WARNING: Unexpected error saving to database: {e}")
        import traceback
        traceback.print_exc()

def load_readings_from_db(experiment_id=None, lights_on=9, exp_start=None):
    """Load readings from database and return as DataFrame matching the original format."""
    if not USE_DATABASE or not DB_AVAILABLE:
        return None
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Get experiment info if not provided
        if experiment_id and (exp_start is None or lights_on == 9):
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT start_date, lights_on_hour FROM experiments WHERE experiment_id = %s",
                        (experiment_id,)
                    )
                    result = cur.fetchone()
                    if result:
                        if exp_start is None:
                            exp_start = result[0]
                        if lights_on == 9:
                            lights_on = result[1] if result[1] else 9
        
        query = """
            SELECT 
                r.datetime,
                r.fly_id,
                r.reading_type as reading,
                r.value,
                r.monitor,
                f.channel,
                f.genotype,
                f.sex,
                f.treatment
            FROM readings r
            JOIN flies f ON r.fly_id = f.fly_id AND r.experiment_id = f.experiment_id
        """
        
        if experiment_id:
            query += f" WHERE r.experiment_id = {experiment_id}"
        
        query += " ORDER BY r.datetime, r.monitor, f.channel, r.reading_type"
        
        df = pd.read_sql(query, engine)
        
        # Add Date, Time, ZT, Phase, Exp_Day columns
        if 'datetime' in df.columns and len(df) > 0:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
            
            # Calculate ZT and Phase
            zt, phase = calculate_zt_phase(df['datetime'], lights_on)
            df['zt'] = zt
            df['phase'] = phase
            
            # Calculate exp_day if exp_start is provided
            if exp_start:
                if isinstance(exp_start, str):
                    exp_start = pd.to_datetime(exp_start).date()
                df['exp_day'] = calculate_exp_day_global(df, exp_start)
        
        engine.dispose()
        return df
    except psycopg2.Error as e:
        print(f"Database error loading readings: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading readings: {e}")
        return None

def load_health_report_from_db(experiment_id=None):
    """Load health report from database."""
    if not USE_DATABASE or not DB_AVAILABLE:
        return None
    
    try:
        engine = create_engine(DATABASE_URL)
        
        query = """
            SELECT 
                hr.fly_id,
                f.monitor,
                f.channel,
                f.genotype,
                f.sex,
                f.treatment,
                hr.status as final_status,
                hr.report_date
            FROM health_reports hr
            JOIN flies f ON hr.fly_id = f.fly_id AND hr.experiment_id = f.experiment_id
        """
        
        if experiment_id:
            query += f" WHERE hr.experiment_id = {experiment_id}"
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except psycopg2.Error as e:
        print(f"Database error loading health report: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading health report: {e}")
        return None

def get_latest_experiment_id():
    """Get the most recent experiment_id."""
    if not USE_DATABASE or not DB_AVAILABLE:
        return None
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1")
                result = cur.fetchone()
                return result[0] if result else None
    except psycopg2.Error as e:
        print(f"Database error getting latest experiment_id: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting latest experiment_id: {e}")
        return None


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def prepare_data_and_health(
    dam_files=None,
    meta_path=None,
    lights_on=DEFAULT_LIGHTS_ON,
    lights_off=DEFAULT_LIGHTS_OFF,
    apply_date_filter_flag=DEFAULT_APPLY_DATE_FILTER,
    exp_start=DEFAULT_EXP_START,
    exp_end=DEFAULT_EXP_END,
    bin_length_min=DEFAULT_BIN_LENGTH_MIN,
    exclude_days=DEFAULT_EXCLUDE_DAYS,
    ref_day=DEFAULT_REF_DAY,
    decline_threshold=DEFAULT_DECLINE_THRESHOLD,
    death_threshold=DEFAULT_DEATH_THRESHOLD,
    transition_window=DEFAULT_TRANSITION_WINDOW
):
    """
    Main function to prepare data and generate health report.
    
    Args:
        dam_files: List of Monitor*.txt file paths
        meta_path: Path to details.txt metadata file
        lights_on: Hour when lights turn on
        lights_off: Hour when lights turn off
        apply_date_filter_flag: Whether to apply date filtering
        exp_start: Experiment start date (None = auto-detect)
        exp_end: Experiment end date (None = auto-detect)
        bin_length_min: Length of each time bin in minutes
        exclude_days: List of days to exclude from health analysis
        ref_day: Reference day for normalization
        decline_threshold: Threshold for unhealthy classification
        death_threshold: Threshold for death classification
        transition_window: Window in minutes around light transitions
        
    Returns:
        tuple: (prepared_data, health_report)
    """
    # Set defaults
    if dam_files is None:
        dam_files = DEFAULT_DAM_FILES
    if meta_path is None:
        meta_path = DEFAULT_META_PATH
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dam_files = [os.path.join(script_dir, f) if not os.path.isabs(f) else f for f in dam_files]
    meta_path = os.path.join(script_dir, meta_path) if not os.path.isabs(meta_path) else meta_path
    
    # ============================================================
    # PART 1: DATA PREPARATION
    # ============================================================
    
    # STEP 1.1: Load and merge data
    # Parse metadata
    fly_metadata = parse_details(meta_path)
    
    # Parse monitor files
    print(f"\n📁 Loading {len(dam_files)} monitor files...")
    time_series_list = []
    for i, dam_file in enumerate(dam_files, 1):
        if not os.path.exists(dam_file):
            sys.exit(1)
        
        # Extract monitor identifier from filename
        # Extract everything after "Monitor" (e.g., "51_06_20_25" from "Monitor51_06_20_25.txt")
        filename = Path(dam_file).stem
        if filename.startswith('Monitor'):
            monitor_num = filename[7:]  # Remove "Monitor" prefix (7 chars), keep rest as string
        else:
            # Fallback: extract all digits (old behavior for backward compatibility)
            monitor_num = ''.join(filter(str.isdigit, filename))
        
        print(f"  [{i}/{len(dam_files)}] Loading {filename}...", end='', flush=True)
        monitor_data = parse_monitor_file(dam_file, monitor_num)
        print(f" ✓ ({len(monitor_data):,} rows)")
        time_series_list.append(monitor_data)
    
    # Combine time-series data
    print(f"\n🔄 Combining monitor data...", end='', flush=True)
    time_series_data = pd.concat(time_series_list, ignore_index=True)
    time_series_data = time_series_data.sort_values(['datetime', 'monitor', 'channel', 'reading']).reset_index(drop=True)
    print(f" ✓ ({len(time_series_data):,} total rows)")
    
    # Merge with metadata
    print(f"🔄 Merging with metadata...", end='', flush=True)
    dam_merged = time_series_data.merge(fly_metadata, on=['monitor', 'channel'])
    print(f" ✓ ({len(dam_merged):,} total rows)")
    
    # Reorder columns
    final_columns = ['datetime', 'monitor', 'channel', 'reading', 'value', 'fly_id', 'genotype', 'sex', 'treatment']
    dam_merged = dam_merged[final_columns]
    
    # STEP 1.2: Calculate time variables
    print(f"🔄 Calculating time variables...", end='', flush=True)
    dam_merged['date'] = pd.to_datetime(dam_merged['datetime']).dt.date
    dam_merged['time'] = pd.to_datetime(dam_merged['datetime']).dt.strftime('%H:%M:%S')
    
    zt, phase = calculate_zt_phase(dam_merged['datetime'], lights_on)
    dam_merged['zt'] = zt
    dam_merged['phase'] = phase
    print(f" ✓ ({len(dam_merged):,} total rows)")  
    
    # STEP 1.3: Optional date filtering
    print(f"🔄 Applying date filters...", end='', flush=True)
    dam_merged, actual_exp_start, actual_exp_end = apply_date_filter(
        dam_merged, apply_date_filter_flag, exp_start, exp_end
    )
    print(f" ✓ ({len(dam_merged):,} rows after filtering)")
    
    # STEP 1.4: Calculate exp_day
    # Ensure actual_exp_start is not None (use minimum date from data if needed)
    if actual_exp_start is None:
        actual_exp_start = pd.to_datetime(dam_merged['datetime']).dt.date.min()
    dam_merged['exp_day'] = calculate_exp_day_global(dam_merged, actual_exp_start)
    
    # STEP 1.5: Prepare data for output
    # Reorder columns for final output
    output_columns = ['datetime', 'date', 'time', 'monitor', 'channel', 'reading', 'value', 
                      'fly_id', 'genotype', 'sex', 'treatment', 'zt', 'phase', 'exp_day']
    dam_merged = dam_merged[output_columns]
    
    # ============================================================
    # PART 2: HEALTH REPORT GENERATION (using in-memory data)
    # ============================================================
    print(f"\n📊 Generating health report...")
    
    # Update thresholds based on bin_length_min
    thresholds = {
        "A1": 12 * 60 / bin_length_min,
        "A2": 24 * 60 / bin_length_min,
        "ACTIVITY_LOW": THRESHOLDS["ACTIVITY_LOW"],
        "INDEX_LOW": THRESHOLDS["INDEX_LOW"],
        "SLEEP_MAX": THRESHOLDS["SLEEP_MAX"],
        "SLEEP_BOUT": THRESHOLDS["SLEEP_BOUT"],
        "MISSING_MAX": THRESHOLDS["MISSING_MAX"]
    }
    
    # STEP 2.1: Prepare data for health analysis
    print(f"  Preparing data for health analysis...", end='', flush=True)
    dam_activity = prep_data_for_health(dam_merged, exclude_days, bin_length_min)
    print(f" ✓")
    
    # STEP 2.2: Calculate daily metrics
    print(f"  Calculating daily metrics...", end='', flush=True)
    daily_summary = calculate_daily_metrics(dam_activity, bin_length_min)
    print(f" ✓")
    
    # STEP 2.3: Normalize to reference day
    daily_summary = normalize_to_ref_day(daily_summary, ref_day, decline_threshold, death_threshold)
    
    # STEP 2.4: Startle test
    print(f"  Running startle test...", end='', flush=True)
    transition_data = startle_test(dam_activity, lights_on, lights_off, transition_window)
    print(f" ✓")
    
    # STEP 2.5: Classify status
    print(f"  Classifying fly status...", end='', flush=True)
    fly_status = classify_status(daily_summary, transition_data, thresholds)
    print(f" ✓")
    
    # STEP 2.6: Apply irreversible death
    print(f"  Applying irreversible death logic...", end='', flush=True)
    fly_status = apply_irreversible_death(fly_status)
    print(f" ✓")
    
    # STEP 2.7: Generate summary
    print(f"  Generating summary...", end='', flush=True)
    health_report = generate_summary(fly_status)
    print(f" ✓")
    
    # STEP 2.8: Save to database
    experiment_id = None
    if USE_DATABASE and DB_AVAILABLE:
        print(f"\n💾 Saving to database...")
        try:
            # Create experiment
            print(f"  Creating experiment...", end='', flush=True)
            experiment_name = f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_id = create_experiment(
                name=experiment_name,
                start_date=actual_exp_start,
                end_date=actual_exp_end,
                lights_on=lights_on,
                lights_off=lights_off
            )
            print(f" ✓ (ID: {experiment_id})")
            
            if experiment_id:
                excluded_db = fly_ids_without_usable_mt_or_pn(dam_merged)
                if excluded_db:
                    n_before = dam_merged['fly_id'].nunique()
                    print(
                        f"  Excluding {len(excluded_db)} flies (no usable MT/Pn: all-zero or missing type) "
                        f"from database..."
                    , end='', flush=True)
                    dam_merged = dam_merged[~dam_merged['fly_id'].isin(excluded_db)].copy()
                    f_sid = _fly_id_col_from_monitor_channel(
                        fly_status['monitor'], fly_status['channel']
                    )
                    fly_status = fly_status[~f_sid.isin(excluded_db)].copy()
                    hr_fid = _fly_id_col_from_monitor_channel(
                        health_report['monitor'], health_report['channel']
                    )
                    health_report = health_report[~hr_fid.isin(excluded_db)].copy()
                    n_after = dam_merged['fly_id'].nunique()
                    print(f" ✓ ({n_before} → {n_after} flies)")
                    if n_after == 0:
                        raise RuntimeError(
                            "All flies were excluded from DB (no non-zero MT and non-zero Pn). "
                            "Check monitor files and details.txt."
                        )

                save_to_database(dam_merged, health_report, fly_status, experiment_id, actual_exp_start)
        except psycopg2.Error as e:
            # Raise error if database is required
            raise RuntimeError(f"Database error saving to database: {e}")
        except Exception as e:
            # Raise error if database is required
            raise RuntimeError(f"Unexpected error saving to database: {e}")
    
    # Store experiment_id in the returned data for use by subsequent steps
    if experiment_id:
        dam_merged.attrs['experiment_id'] = experiment_id
        if health_report is not None:
            health_report.attrs['experiment_id'] = experiment_id
    
    return dam_merged, health_report



# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 1: Prepare data and generate health report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data input
    parser.add_argument('--dam-files', nargs='+', default=None,
                       help='List of Monitor*.txt files (default: Monitor5.txt, Monitor6.txt)')
    parser.add_argument('--meta-path', type=str, default=None,
                       help='Metadata file path (default: details.txt)')
    
    # Settings
    parser.add_argument('--lights-on', type=int, default=DEFAULT_LIGHTS_ON,
                       help=f'Hour when lights turn on (default: {DEFAULT_LIGHTS_ON})')
    parser.add_argument('--lights-off', type=int, default=DEFAULT_LIGHTS_OFF,
                       help=f'Hour when lights turn off (default: {DEFAULT_LIGHTS_OFF})')
    
    # Date filtering
    parser.add_argument('--apply-date-filter', action='store_true',
                       help='Enable date filtering')
    parser.add_argument('--exp-start', type=str, default=None,
                       help='Experiment start date (YYYY-MM-DD, default: auto-detect)')
    parser.add_argument('--exp-end', type=str, default=None,
                       help='Experiment end date (YYYY-MM-DD, default: auto-detect)')
    
    # Health report settings
    parser.add_argument('--ref-day', type=int, default=DEFAULT_REF_DAY,
                       help=f'Reference day for normalization (default: {DEFAULT_REF_DAY})')
    parser.add_argument('--exclude-days', nargs='+', type=int, default=DEFAULT_EXCLUDE_DAYS,
                       help=f'Days to exclude from health analysis (default: {DEFAULT_EXCLUDE_DAYS})')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    exp_start = pd.to_datetime(args.exp_start).date() if args.exp_start else None
    exp_end = pd.to_datetime(args.exp_end).date() if args.exp_end else None
    
    # Run pipeline
    prepare_data_and_health(
        dam_files=args.dam_files,
        meta_path=args.meta_path,
        lights_on=args.lights_on,
        lights_off=args.lights_off,
        apply_date_filter_flag=args.apply_date_filter,
        exp_start=exp_start,
        exp_end=exp_end,
        exclude_days=args.exclude_days,
        ref_day=args.ref_day
    )


if __name__ == "__main__":
    main()


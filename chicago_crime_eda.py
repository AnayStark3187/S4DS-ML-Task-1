#!/usr/bin/env python3
"""
Chicago Crime Dataset — Comprehensive Exploratory Data Analysis
================================================================

A complete EDA pipeline analyzing the Chicago crime dataset to discover
patterns in crime type, location, time, arrests, and other variables.

Author: Data Analysis Pipeline
Date: 2026-08-15
"""

# ============================================================================
# 1. IMPORT LIBRARIES & ENVIRONMENT CHECK
# ============================================================================

import os
import sys
import glob
import warnings

# Auto-detect virtual environment if third-party packages are missing
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, '.venv', 'bin', 'python3')
    if os.path.exists(venv_python) and sys.executable != os.path.abspath(venv_python):
        print(f"Redirecting execution to project virtual environment: {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("ERROR: Required dependencies (numpy, pandas, matplotlib, seaborn, scipy) are missing.")
        print("Please activate your virtual environment or install dependencies via:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

warnings.filterwarnings('ignore')

# Global plot settings
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
PALETTE = sns.color_palette("husl", 15)
ACCENT = '#2196F3'
ACCENT2 = '#FF5722'
SEQUENTIAL_CMAP = 'YlOrRd'

# Output directory for plots
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def find_dataset():
    """Auto-detect the Chicago Crime dataset file in the project directory."""
    extensions = ['*.csv', '*.CSV', '*.xlsx', '*.xls', '*.parquet', '*.json']
    candidates = []
    for ext in extensions:
        candidates.extend(glob.glob(ext))
        candidates.extend(glob.glob(os.path.join('data', ext)))
    
    if not candidates:
        print("=" * 70)
        print("DATASET NOT FOUND")
        print("=" * 70)
        print("\nPlease place the Chicago Crime dataset file in this directory.")
        print("Expected file: a CSV file (e.g., 'chicago_crime.csv')")
        print("\nDownload from:")
        print("  https://drive.google.com/file/d/1Efog2t2MWgm1Ciyn2nZyXqR9YAhUm7Ur/view")
        print("\nSupported formats: CSV, Excel, Parquet, JSON")
        print("=" * 70)
        return None
    
    # Prefer files with 'crime' or 'chicago' in the name
    for c in candidates:
        if 'crime' in c.lower() or 'chicago' in c.lower():
            return c
    return candidates[0]


def load_dataset(filepath):
    """Load dataset based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    print(f"Loading dataset from: {filepath}")
    
    if ext == '.csv':
        df = pd.read_csv(filepath, low_memory=False)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def detect_columns(df):
    """Create a flexible column mapping for common Chicago Crime dataset fields."""
    col_map = {}
    columns_lower = {c.lower().replace(' ', '_').replace('.', ''): c for c in df.columns}
    
    mappings = {
        'id': ['id', 'case_number', 'case_id'],
        'case_number': ['case_number', 'case_id'],
        'date': ['date', 'datetime', 'date_time', 'occurred_date'],
        'block': ['block', 'address'],
        'iucr': ['iucr', 'iucr_code'],
        'primary_type': ['primary_type', 'primarytype', 'crime_type', 'type'],
        'description': ['description', 'crime_description', 'desc'],
        'location_description': ['location_description', 'locationdescription', 'location_desc'],
        'arrest': ['arrest', 'arrested'],
        'domestic': ['domestic', 'is_domestic'],
        'beat': ['beat'],
        'district': ['district'],
        'ward': ['ward'],
        'community_area': ['community_area', 'communityarea', 'community'],
        'fbi_code': ['fbi_code', 'fbicode', 'fbi'],
        'x_coordinate': ['x_coordinate', 'xcoordinate', 'x'],
        'y_coordinate': ['y_coordinate', 'ycoordinate', 'y'],
        'year': ['year'],
        'latitude': ['latitude', 'lat'],
        'longitude': ['longitude', 'lng', 'lon', 'long'],
        'location': ['location', 'loc'],
        'updated_on': ['updated_on', 'updatedon'],
    }
    
    for key, possible_names in mappings.items():
        for name in possible_names:
            if name in columns_lower:
                col_map[key] = columns_lower[name]
                break
    
    return col_map


def plot_top_categories(series, n=15, title='', xlabel='Count', ylabel='',
                        color=ACCENT, figsize=(12, 7), horizontal=True,
                        save_name=None):
    """Plot top N categories from a Series."""
    top = series.value_counts().head(n)
    
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        top.sort_values().plot(kind='barh', ax=ax, color=color, edgecolor='white')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i, v in enumerate(top.sort_values()):
            ax.text(v + top.max() * 0.01, i, f'{v:,}', va='center', fontsize=9)
    else:
        top.plot(kind='bar', ax=ax, color=color, edgecolor='white')
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(PLOT_DIR, save_name), bbox_inches='tight', dpi=150)
    plt.show()


def plot_percentage_bar(labels, values, title='', xlabel='', ylabel='',
                        color=ACCENT, figsize=(12, 7), save_name=None):
    """Plot horizontal bar chart with percentage values."""
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=color, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(PLOT_DIR, save_name), bbox_inches='tight', dpi=150)
    plt.show()


def calculate_percentage(series):
    """Calculate value counts as percentages."""
    counts = series.value_counts()
    percentages = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({'Count': counts, 'Percentage': percentages})


def separator(title):
    """Print a formatted section separator."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ============================================================================
# 3. LOAD DATASET
# ============================================================================

def main():
    """Main EDA pipeline."""
    
    separator("CHICAGO CRIME DATASET — EXPLORATORY DATA ANALYSIS")
    
    filepath = find_dataset()
    if filepath is None:
        return
    
    df = load_dataset(filepath)
    col_map = detect_columns(df)
    
    print(f"\nDetected column mapping:")
    for key, col in col_map.items():
        print(f"  {key:25s} → {col}")
    
    # ========================================================================
    # 4. DATA OVERVIEW
    # ========================================================================
    separator("4. DATA OVERVIEW")
    
    print("--- First 5 Rows ---")
    print(df.head().to_string())
    
    print("\n--- Last 5 Rows ---")
    print(df.tail().to_string())
    
    print(f"\n--- Dataset Shape ---")
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    print(f"\n--- Column Names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n--- Data Types ---")
    print(df.dtypes.to_string())
    
    print(f"\n--- Memory Usage ---")
    mem = df.memory_usage(deep=True)
    print(f"Total: {mem.sum() / 1e6:.1f} MB")
    print(mem.to_string())
    
    print(f"\n--- Numerical Summary Statistics ---")
    print(df.describe().to_string())
    
    obj_cols = df.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        print(f"\n--- Categorical Summary Statistics ---")
        print(df[obj_cols].describe().to_string())
    
    # ========================================================================
    # 5. DATA UNDERSTANDING — Variable Classification
    # ========================================================================
    separator("5. DATA UNDERSTANDING")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    print(f"Numerical Variables ({len(numerical_cols)}):")
    for c in numerical_cols:
        print(f"  - {c}")
    
    print(f"\nCategorical Variables ({len(categorical_cols)}):")
    for c in categorical_cols:
        print(f"  - {c}")
    
    # --- Parse Date Column ---
    date_col = col_map.get('date')
    if date_col and date_col in df.columns:
        print(f"\nParsing date column: '{date_col}'")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract temporal features
        df['Year_extracted'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Month_Name'] = df[date_col].dt.month_name()
        df['Day'] = df[date_col].dt.day
        df['Day_of_Week'] = df[date_col].dt.day_name()
        df['Day_of_Week_Num'] = df[date_col].dt.dayofweek
        df['Hour'] = df[date_col].dt.hour
        df['Quarter'] = df[date_col].dt.quarter
        df['Is_Weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        print(f"  Extracted: Year, Month, Day, Day_of_Week, Hour, Quarter, Is_Weekend")
        print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    # Ensure 'Year' column exists (from dataset or extracted)
    year_col = col_map.get('year', 'Year_extracted')
    if year_col not in df.columns and 'Year_extracted' in df.columns:
        year_col = 'Year_extracted'
    
    # ========================================================================
    # 6. DATA QUALITY ASSESSMENT
    # ========================================================================
    separator("6. DATA QUALITY ASSESSMENT")
    
    # --- 6.1 Missing Values ---
    print("--- 6.1 Missing Values ---\n")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Percentage', ascending=False)
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
        
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.4)))
        colors = ['#FF5722' if p > 10 else '#FFC107' if p > 1 else '#4CAF50'
                  for p in missing_df['Missing_Percentage']]
        missing_df['Missing_Percentage'].sort_values().plot(
            kind='barh', ax=ax, color=colors[::-1], edgecolor='white'
        )
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        for i, (idx, row) in enumerate(missing_df.sort_values('Missing_Percentage').iterrows()):
            ax.text(row['Missing_Percentage'] + 0.2, i,
                    f"{row['Missing_Percentage']:.1f}% ({row['Missing_Count']:,})",
                    va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '01_missing_values.png'), bbox_inches='tight', dpi=150)
        plt.show()
    else:
        print("No missing values found in the dataset!")
    
    # --- 6.2 Duplicate Records ---
    print("\n--- 6.2 Duplicate Records ---\n")
    n_dupes = df.duplicated().sum()
    pct_dupes = n_dupes / len(df) * 100
    print(f"Completely duplicated rows: {n_dupes:,} ({pct_dupes:.2f}%)")
    
    id_col = col_map.get('id')
    if id_col and id_col in df.columns:
        n_unique_ids = df[id_col].nunique()
        n_total = len(df)
        print(f"Unique IDs ({id_col}): {n_unique_ids:,} out of {n_total:,}")
        if n_unique_ids < n_total:
            print(f"  → {n_total - n_unique_ids:,} records share an ID with another record.")
    
    case_col = col_map.get('case_number')
    if case_col and case_col in df.columns and case_col != id_col:
        n_unique_cases = df[case_col].nunique()
        print(f"Unique Case Numbers ({case_col}): {n_unique_cases:,}")
    
    # --- 6.3 Inconsistent Values ---
    print("\n--- 6.3 Inconsistent Values ---\n")
    for c in categorical_cols[:10]:  # Check top categorical columns
        if c in df.columns:
            unique_vals = df[c].dropna().unique()
            # Check for whitespace issues
            stripped = pd.Series(unique_vals).astype(str).str.strip()
            whitespace_issues = (stripped != pd.Series(unique_vals).astype(str)).sum()
            if whitespace_issues > 0:
                print(f"  {c}: {whitespace_issues} values with leading/trailing spaces")
            
            # Check for null-like strings
            null_strings = ['N/A', 'NA', 'None', 'null', 'NULL', 'Unknown', 'UNKNOWN', '']
            found_nulls = [v for v in unique_vals if str(v).strip() in null_strings]
            if found_nulls:
                print(f"  {c}: Found null-like strings: {found_nulls}")
            
            # Report rare categories (< 0.1% of data)
            vc = df[c].value_counts()
            rare = vc[vc / len(df) < 0.001]
            if len(rare) > 0 and len(rare) <= 10:
                print(f"  {c}: {len(rare)} rare categories (< 0.1% each): {list(rare.index)}")
            elif len(rare) > 10:
                print(f"  {c}: {len(rare)} rare categories (< 0.1% each)")
    
    # --- 6.4 Numerical Anomalies ---
    print("\n--- 6.4 Numerical Anomalies ---\n")
    
    lat_col = col_map.get('latitude')
    lon_col = col_map.get('longitude')
    
    if lat_col and lat_col in df.columns:
        lat_valid = df[lat_col].dropna()
        # Chicago approximate boundaries: lat 41.6–42.1, lon -87.9 to -87.5
        out_of_bounds = lat_valid[(lat_valid < 41.6) | (lat_valid > 42.1)]
        print(f"Latitude: {len(out_of_bounds):,} values outside Chicago bounds (41.6–42.1)")
        if len(out_of_bounds) > 0:
            print(f"  Range: {lat_valid.min():.4f} to {lat_valid.max():.4f}")
    
    if lon_col and lon_col in df.columns:
        lon_valid = df[lon_col].dropna()
        out_of_bounds = lon_valid[(lon_valid < -87.95) | (lon_valid > -87.5)]
        print(f"Longitude: {len(out_of_bounds):,} values outside Chicago bounds (-87.95 to -87.5)")
        if len(out_of_bounds) > 0:
            print(f"  Range: {lon_valid.min():.4f} to {lon_valid.max():.4f}")
    
    # Box plots for key numerical variables
    num_for_box = [c for c in ['District', 'Ward', 'Community Area', 'Beat']
                   if c in df.columns or col_map.get(c.lower().replace(' ', '_'), '') in df.columns]
    actual_num_cols = []
    for c in num_for_box:
        actual = col_map.get(c.lower().replace(' ', '_'), c)
        if actual in df.columns and pd.api.types.is_numeric_dtype(df[actual]):
            actual_num_cols.append(actual)
    
    if actual_num_cols:
        fig, axes = plt.subplots(1, len(actual_num_cols),
                                 figsize=(4 * len(actual_num_cols), 5))
        if len(actual_num_cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, actual_num_cols):
            df[c].dropna().plot(kind='box', ax=ax)
            ax.set_title(c, fontsize=12)
        plt.suptitle('Numerical Variable Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '02_numerical_boxplots.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # ========================================================================
    # 7. UNIVARIATE ANALYSIS
    # ========================================================================
    separator("7. UNIVARIATE ANALYSIS")
    
    # --- 7.1 Crime Type ---
    print("--- 7.1 Crime Type Distribution ---\n")
    pt_col = col_map.get('primary_type')
    if pt_col and pt_col in df.columns:
        crime_counts = df[pt_col].value_counts()
        print(f"Total unique crime types: {len(crime_counts)}")
        print(f"\nTop 10 Crime Types:")
        print(crime_counts.head(10).to_string())
        print(f"\nBottom 5 Crime Types:")
        print(crime_counts.tail(5).to_string())
        
        plot_top_categories(
            df[pt_col], n=15,
            title='Top 15 Crime Types in Chicago',
            xlabel='Number of Incidents',
            save_name='03_crime_types.png'
        )
    
    # --- 7.2 Crime Description ---
    print("\n--- 7.2 Crime Description Distribution ---\n")
    desc_col = col_map.get('description')
    if desc_col and desc_col in df.columns:
        print(f"Total unique descriptions: {df[desc_col].nunique()}")
        plot_top_categories(
            df[desc_col], n=15,
            title='Top 15 Crime Descriptions',
            xlabel='Number of Incidents',
            color='#FF9800',
            save_name='04_crime_descriptions.png'
        )
    
    # --- 7.3 Arrests ---
    print("\n--- 7.3 Arrest Distribution ---\n")
    arr_col = col_map.get('arrest')
    if arr_col and arr_col in df.columns:
        arrest_dist = calculate_percentage(df[arr_col])
        print(arrest_dist.to_string())
        
        overall_arrest_rate = pd.to_numeric(df[arr_col], errors='coerce').mean() * 100
        print(f"\nOverall Arrest Rate: {overall_arrest_rate:.1f}%")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        arrest_counts = df[arr_col].value_counts()
        labels = ['No Arrest', 'Arrest'] if arrest_counts.index[0] in [False, 0] \
            else [str(arrest_counts.index[0]), str(arrest_counts.index[1])]
        colors_arr = ['#EF5350', '#66BB6A']
        ax1.bar(labels, arrest_counts.values, color=colors_arr, edgecolor='white', width=0.5)
        ax1.set_title('Arrest Distribution (Count)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Incidents')
        for i, v in enumerate(arrest_counts.values):
            ax1.text(i, v + arrest_counts.max() * 0.02, f'{v:,}', ha='center', fontsize=11)
        
        # Pie chart
        ax2.pie(arrest_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors_arr, startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Arrest Distribution (Percentage)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '05_arrest_distribution.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- 7.4 Domestic Crimes ---
    print("\n--- 7.4 Domestic Crime Distribution ---\n")
    dom_col = col_map.get('domestic')
    if dom_col and dom_col in df.columns:
        dom_dist = calculate_percentage(df[dom_col])
        print(dom_dist.to_string())
        
        domestic_rate = pd.to_numeric(df[dom_col], errors='coerce').mean() * 100
        print(f"\nDomestic Crime Rate: {domestic_rate:.1f}%")
    
    # --- 7.5 Location ---
    print("\n--- 7.5 Location Distribution ---\n")
    loc_desc_col = col_map.get('location_description')
    if loc_desc_col and loc_desc_col in df.columns:
        print(f"Unique location types: {df[loc_desc_col].nunique()}")
        plot_top_categories(
            df[loc_desc_col], n=15,
            title='Top 15 Crime Locations',
            xlabel='Number of Incidents',
            color='#9C27B0',
            save_name='06_crime_locations.png'
        )
    
    dist_col = col_map.get('district')
    if dist_col and dist_col in df.columns:
        print(f"\nCrimes by District:")
        dist_counts = df[dist_col].value_counts().head(15)
        print(dist_counts.to_string())
    
    # --- 7.6 Temporal Distribution ---
    print("\n--- 7.6 Temporal Distribution ---\n")
    
    # Crimes by Year
    if year_col in df.columns:
        year_counts = df[year_col].value_counts().sort_index()
        print(f"Crimes by Year:")
        print(year_counts.to_string())
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2.5,
                color=ACCENT, markersize=7)
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color=ACCENT)
        ax.set_title('Crime Incidents by Year', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Incidents')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '07_crimes_by_year.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # Crimes by Month
    if 'Month_Name' in df.columns:
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_counts = df['Month_Name'].value_counts().reindex(month_order)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(range(len(month_counts)), month_counts.values,
                      color=sns.color_palette("coolwarm", 12), edgecolor='white')
        ax.set_xticks(range(12))
        ax.set_xticklabels([m[:3] for m in month_order], fontsize=11)
        ax.set_title('Crime Incidents by Month', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Incidents')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '08_crimes_by_month.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # Crimes by Day of Week
    if 'Day_of_Week' in df.columns:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df['Day_of_Week'].value_counts().reindex(dow_order)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_dow = ['#42A5F5' if d in ['Saturday', 'Sunday'] else '#78909C' for d in dow_order]
        ax.bar(dow_order, dow_counts.values, color=colors_dow, edgecolor='white')
        ax.set_title('Crime Incidents by Day of Week', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Incidents')
        for i, v in enumerate(dow_counts.values):
            ax.text(i, v + dow_counts.max() * 0.01, f'{v:,}', ha='center', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '09_crimes_by_day.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # Crimes by Hour
    if 'Hour' in df.columns:
        hour_counts = df['Hour'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(14, 5))
        colors_hour = ['#FFA726' if 18 <= h or h <= 5 else '#42A5F5' for h in range(24)]
        ax.bar(hour_counts.index, hour_counts.values, color=colors_hour, edgecolor='white')
        ax.set_title('Crime Incidents by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour (0–23)')
        ax.set_ylabel('Number of Incidents')
        ax.set_xticks(range(24))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '10_crimes_by_hour.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # ========================================================================
    # 8. BIVARIATE ANALYSIS
    # ========================================================================
    separator("8. BIVARIATE ANALYSIS")
    
    # --- 8.1 Crime Type vs Arrest ---
    print("--- 8.1 Crime Type vs Arrest ---\n")
    if pt_col and arr_col and pt_col in df.columns and arr_col in df.columns:
        top_crimes = df[pt_col].value_counts().head(15).index
        arrest_by_type = df[df[pt_col].isin(top_crimes)].groupby(pt_col)[arr_col].mean() * 100
        arrest_by_type = arrest_by_type.sort_values(ascending=True)
        
        print("Arrest Rate by Crime Type (Top 15):")
        for crime, rate in arrest_by_type.items():
            count = df[df[pt_col] == crime].shape[0]
            print(f"  {crime:35s}  {rate:6.1f}%  (n={count:,})")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors_arrest = ['#66BB6A' if r > 50 else '#FFA726' if r > 25 else '#EF5350'
                         for r in arrest_by_type.values]
        ax.barh(range(len(arrest_by_type)), arrest_by_type.values,
                color=colors_arrest, edgecolor='white')
        ax.set_yticks(range(len(arrest_by_type)))
        ax.set_yticklabels(arrest_by_type.index)
        ax.set_xlabel('Arrest Rate (%)')
        ax.set_title('Arrest Rate by Crime Type (Top 15 Crime Types)',
                      fontsize=14, fontweight='bold')
        ax.axvline(x=overall_arrest_rate, color='red', linestyle='--', alpha=0.7,
                   label=f'Overall Rate: {overall_arrest_rate:.1f}%')
        ax.legend(loc='lower right')
        for i, v in enumerate(arrest_by_type.values):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '11_arrest_rate_by_crime.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- 8.2 Crime Type vs Time ---
    print("\n--- 8.2 Crime Type vs Time ---\n")
    if pt_col and year_col and pt_col in df.columns and year_col in df.columns:
        top5_crimes = df[pt_col].value_counts().head(5).index
        crime_year = df[df[pt_col].isin(top5_crimes)].groupby(
            [year_col, pt_col]).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, crime in enumerate(top5_crimes):
            if crime in crime_year.columns:
                ax.plot(crime_year.index, crime_year[crime], marker='o',
                        label=crime, linewidth=2, markersize=5)
        ax.set_title('Top 5 Crime Types Over Years', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Incidents')
        ax.legend(loc='best', framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '12_crime_trends_by_year.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # Crime Type x Month heatmap
    if pt_col and 'Month' in df.columns:
        top8_crimes = df[pt_col].value_counts().head(8).index
        crime_month = df[df[pt_col].isin(top8_crimes)].groupby(
            ['Month', pt_col]).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(crime_month.T, annot=True, fmt='d', cmap='YlOrRd',
                    ax=ax, linewidths=0.5)
        ax.set_title('Crime Counts: Crime Type × Month', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Crime Type')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '13_crime_type_month_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- 8.3 Crime Type vs Location ---
    print("\n--- 8.3 Crime Type vs Location ---\n")
    if pt_col and loc_desc_col and pt_col in df.columns and loc_desc_col in df.columns:
        top8_crimes = df[pt_col].value_counts().head(8).index
        top8_locs = df[loc_desc_col].value_counts().head(8).index
        
        crime_loc = df[df[pt_col].isin(top8_crimes) & df[loc_desc_col].isin(top8_locs)] \
            .groupby([pt_col, loc_desc_col]).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(crime_loc, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5)
        ax.set_title('Crime Type × Location (Top 8 each)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Location')
        ax.set_ylabel('Crime Type')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '14_crime_type_location_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- 8.4 Crime vs Day of Week ---
    print("\n--- 8.4 Crime vs Day of Week ---\n")
    if 'Day_of_Week' in df.columns:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df['Day_of_Week'].value_counts().reindex(dow_order)
        print(f"Highest crime day: {dow_counts.idxmax()} ({dow_counts.max():,})")
        print(f"Lowest crime day:  {dow_counts.idxmin()} ({dow_counts.min():,})")
    
    # --- 8.5 Crime vs Hour ---
    print("\n--- 8.5 Crime vs Hour ---\n")
    if 'Hour' in df.columns:
        hour_counts = df['Hour'].value_counts().sort_index()
        print(f"Peak crime hour:   {hour_counts.idxmax()}:00 ({hour_counts.max():,})")
        print(f"Lowest crime hour: {hour_counts.idxmin()}:00 ({hour_counts.min():,})")
        
        # Crime type by hour for top 5
        if pt_col and pt_col in df.columns:
            top5 = df[pt_col].value_counts().head(5).index
            fig, ax = plt.subplots(figsize=(14, 6))
            for crime in top5:
                subset = df[df[pt_col] == crime]
                hourly = subset['Hour'].value_counts().sort_index()
                ax.plot(hourly.index, hourly.values, label=crime, linewidth=2)
            ax.set_title('Hourly Crime Patterns by Crime Type (Top 5)',
                          fontsize=14, fontweight='bold')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Number of Incidents')
            ax.set_xticks(range(24))
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '15_hourly_crime_by_type.png'), bbox_inches='tight', dpi=150)
            plt.show()
    
    # --- 8.6 Arrest vs Time ---
    print("\n--- 8.6 Arrest vs Time ---\n")
    if arr_col and arr_col in df.columns:
        # Arrest rate by year
        if year_col in df.columns:
            arrest_year = df.groupby(year_col)[arr_col].mean() * 100
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(arrest_year.index, arrest_year.values, marker='o',
                    linewidth=2.5, color=ACCENT2, markersize=7)
            ax.set_title('Arrest Rate Over Years', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Arrest Rate (%)')
            ax.axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '16_arrest_rate_by_year.png'), bbox_inches='tight', dpi=150)
            plt.show()
        
        # Arrest rate by hour
        if 'Hour' in df.columns:
            arrest_hour = df.groupby('Hour')[arr_col].mean() * 100
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(arrest_hour.index, arrest_hour.values, color=ACCENT2, edgecolor='white')
            ax.set_title('Arrest Rate by Hour of Day', fontsize=14, fontweight='bold')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Arrest Rate (%)')
            ax.set_xticks(range(24))
            ax.axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '17_arrest_rate_by_hour.png'), bbox_inches='tight', dpi=150)
            plt.show()
    
    # ========================================================================
    # 9. MULTIVARIATE ANALYSIS
    # ========================================================================
    separator("9. MULTIVARIATE ANALYSIS")
    
    # --- Day of Week × Hour Heatmap ---
    print("--- Day of Week × Hour Crime Heatmap ---\n")
    if 'Day_of_Week_Num' in df.columns and 'Hour' in df.columns:
        dow_hour = df.groupby(['Day_of_Week_Num', 'Hour']).size().unstack(fill_value=0)
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(dow_hour, cmap='YlOrRd', ax=ax, linewidths=0.3,
                    yticklabels=dow_labels, annot=False)
        ax.set_title('Crime Heatmap: Day of Week × Hour', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '18_day_hour_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- Crime Type × Hour × Arrest ---
    print("\n--- Crime Type × Hour × Arrest ---\n")
    if pt_col and arr_col and 'Hour' in df.columns:
        top5 = df[pt_col].value_counts().head(5).index
        subset = df[df[pt_col].isin(top5)]
        
        pivot = subset.groupby([pt_col, 'Hour'])[arr_col].mean().unstack(fill_value=0) * 100
        
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot, cmap='RdYlGn', ax=ax, linewidths=0.3,
                    vmin=0, vmax=100, fmt='.0f')
        ax.set_title('Arrest Rate (%): Crime Type × Hour (Top 5 Crimes)',
                      fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Crime Type')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '19_crime_hour_arrest_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # --- District × Crime Type ---
    print("\n--- District × Crime Type ---\n")
    if dist_col and pt_col and dist_col in df.columns and pt_col in df.columns:
        top8_crimes = df[pt_col].value_counts().head(8).index
        top10_districts = df[dist_col].value_counts().head(10).index
        
        dist_crime = df[df[pt_col].isin(top8_crimes) & df[dist_col].isin(top10_districts)] \
            .groupby([dist_col, pt_col]).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(dist_crime, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5)
        ax.set_title('Crime Distribution: District × Crime Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Crime Type')
        ax.set_ylabel('District')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '20_district_crime_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
    
    # ========================================================================
    # 10. GEOGRAPHIC ANALYSIS
    # ========================================================================
    separator("10. GEOGRAPHIC ANALYSIS")
    
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        # Filter valid coordinates
        geo = df.dropna(subset=[lat_col, lon_col])
        geo = geo[(geo[lat_col] > 41.6) & (geo[lat_col] < 42.1) &
                  (geo[lon_col] > -87.95) & (geo[lon_col] < -87.5)]
        
        print(f"Records with valid coordinates: {len(geo):,} / {len(df):,}")
        
        # Overall crime scatter
        fig, ax = plt.subplots(figsize=(10, 12))
        sample_size = min(50000, len(geo))
        geo_sample = geo.sample(n=sample_size, random_state=42)
        ax.scatter(geo_sample[lon_col], geo_sample[lat_col],
                   s=0.3, alpha=0.05, c=ACCENT)
        ax.set_title(f'Crime Geographic Distribution (n={sample_size:,} sampled)',
                      fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '21_geographic_scatter.png'), bbox_inches='tight', dpi=150)
        plt.show()
        
        # Crime type spatial distribution (top 4)
        if pt_col and pt_col in df.columns:
            top4 = df[pt_col].value_counts().head(4).index
            fig, axes = plt.subplots(2, 2, figsize=(14, 16))
            axes = axes.flatten()
            
            for i, crime in enumerate(top4):
                crime_geo = geo[geo[pt_col] == crime]
                sample_n = min(10000, len(crime_geo))
                if sample_n > 0:
                    crime_sample = crime_geo.sample(n=sample_n, random_state=42)
                    axes[i].scatter(crime_sample[lon_col], crime_sample[lat_col],
                                    s=0.5, alpha=0.1, c=PALETTE[i])
                axes[i].set_title(f'{crime} (n={len(crime_geo):,})', fontsize=12)
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
                axes[i].set_aspect('equal')
            
            plt.suptitle('Spatial Distribution by Crime Type', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '22_geographic_by_crime.png'), bbox_inches='tight', dpi=150)
            plt.show()
    else:
        print("Latitude/Longitude columns not found. Skipping geographic analysis.")
    
    # ========================================================================
    # 11. CORRELATION ANALYSIS
    # ========================================================================
    separator("11. CORRELATION ANALYSIS")
    
    # Select meaningful numerical columns (excluding IDs and duplicate extracted columns)
    exclude_patterns = ['id', 'case', 'x_coord', 'y_coord', 'updated', '_year', 'year_extracted']
    meaningful_num = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if not any(pat in c.lower() for pat in exclude_patterns):
            meaningful_num.append(c)
    
    if len(meaningful_num) >= 2:
        corr = df[meaningful_num].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix (Meaningful Numerical Variables)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '23_correlation_heatmap.png'), bbox_inches='tight', dpi=150)
        plt.show()
        
        # Report meaningful correlations
        print("\nNotable Correlations (|r| > 0.3, excluding diagonal):")
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) > 0.3:
                    print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: r = {r:.3f}")
    else:
        print("Not enough meaningful numerical columns for correlation analysis.")
    
    # ========================================================================
    # 12. STATISTICAL ANALYSIS
    # ========================================================================
    separator("12. STATISTICAL ANALYSIS")
    
    # --- Chi-square: Crime Type × Arrest ---
    print("--- Chi-square Test: Crime Type × Arrest ---\n")
    if pt_col and arr_col and pt_col in df.columns and arr_col in df.columns:
        top10_crimes = df[pt_col].value_counts().head(10).index
        subset = df[df[pt_col].isin(top10_crimes)]
        contingency = pd.crosstab(subset[pt_col], subset[arr_col])
        
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"H0: Arrest outcome is independent of crime type.")
        print(f"H1: Arrest outcome depends on crime type.")
        print(f"\nChi-square statistic: {chi2:,.2f}")
        print(f"Degrees of freedom:   {dof}")
        print(f"P-value:              {p_val:.2e}")
        
        if p_val < 0.05:
            print(f"\n→ REJECT H0 (p < 0.05). There is a statistically significant")
            print(f"  association between crime type and arrest outcome.")
            # Cramér's V for effect size
            n = contingency.sum().sum()
            min_dim = max(1, min(contingency.shape) - 1)
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            print(f"  Cramér's V = {cramers_v:.3f} (effect size)")
            if cramers_v < 0.1:
                print(f"  → Weak association (V < 0.1)")
            elif cramers_v < 0.3:
                print(f"  → Moderate association (0.1 ≤ V < 0.3)")
            else:
                print(f"  → Strong association (V ≥ 0.3)")
        else:
            print(f"\n→ FAIL TO REJECT H0. No significant association found.")
        
        print(f"\nCaveat: Statistical significance with large datasets does not")
        print(f"imply practical significance. The large sample size (n={len(subset):,})")
        print(f"means even small effects can be statistically significant.")
    
    # --- Chi-square: Crime Type × Domestic ---
    if pt_col and dom_col and pt_col in df.columns and dom_col in df.columns:
        print(f"\n--- Chi-square Test: Crime Type × Domestic ---\n")
        top10_crimes = df[pt_col].value_counts().head(10).index
        subset = df[df[pt_col].isin(top10_crimes)]
        contingency2 = pd.crosstab(subset[pt_col], subset[dom_col])
        
        chi2_2, p_val_2, dof_2, _ = stats.chi2_contingency(contingency2)
        n2 = contingency2.sum().sum()
        min_dim2 = max(1, min(contingency2.shape) - 1)
        cramers_v2 = np.sqrt(chi2_2 / (n2 * min_dim2))
        
        print(f"H0: Domestic classification is independent of crime type.")
        print(f"H1: Domestic classification depends on crime type.")
        print(f"\nChi-square statistic: {chi2_2:,.2f}")
        print(f"P-value:              {p_val_2:.2e}")
        print(f"Cramér's V:           {cramers_v2:.3f}")
    
    # --- Weekend vs Weekday crime comparison ---
    if 'Is_Weekend' in df.columns:
        print(f"\n--- Weekend vs Weekday Crime Comparison ---\n")
        weekend_counts = df[df['Is_Weekend'] == 1].shape[0]
        weekday_counts = df[df['Is_Weekend'] == 0].shape[0]
        
        # Normalize by number of days (2 weekend, 5 weekday)
        weekend_daily_avg = weekend_counts / 2
        weekday_daily_avg = weekday_counts / 5
        
        print(f"Weekend total: {weekend_counts:,}")
        print(f"Weekday total: {weekday_counts:,}")
        print(f"Weekend daily average (per day-type): {weekend_daily_avg:,.0f}")
        print(f"Weekday daily average (per day-type): {weekday_daily_avg:,.0f}")
        ratio = weekend_daily_avg / weekday_daily_avg if weekday_daily_avg > 0 else 0
        print(f"Weekend/Weekday ratio: {ratio:.2f}")
    
    # ========================================================================
    # 13. KEY INSIGHTS
    # ========================================================================
    separator("13. KEY INSIGHTS")
    
    insights = []
    
    # Insight 1: Dataset overview
    date_col_actual = col_map.get('date')
    if date_col_actual and date_col_actual in df.columns:
        min_year = df[date_col_actual].dt.year.min()
        max_year = df[date_col_actual].dt.year.max()
        insights.append(
            f"1. DATASET SCOPE: The dataset contains {len(df):,} crime records "
            f"spanning {min_year}–{max_year}, providing a comprehensive view of "
            f"crime in Chicago."
        )
    
    # Insight 2: Top crime
    if pt_col and pt_col in df.columns:
        top_crime = df[pt_col].value_counts().head(1)
        top_crime_name = top_crime.index[0]
        top_crime_pct = top_crime.values[0] / len(df) * 100
        insights.append(
            f"2. DOMINANT CRIME TYPE: '{top_crime_name}' is the most common crime "
            f"category, accounting for {top_crime_pct:.1f}% of all incidents."
        )
    
    # Insight 3: Arrest rate
    if arr_col and arr_col in df.columns:
        insights.append(
            f"3. ARREST RATE: The overall arrest rate is {overall_arrest_rate:.1f}%, "
            f"indicating that a majority of reported crimes do not result in arrest."
        )
    
    # Insight 4: Temporal peaks
    if 'Hour' in df.columns:
        peak_hour = df['Hour'].value_counts().idxmax()
        low_hour = df['Hour'].value_counts().idxmin()
        insights.append(
            f"4. TEMPORAL PATTERNS: Crime peaks at {peak_hour}:00 and is lowest at "
            f"{low_hour}:00, suggesting distinct diurnal patterns in criminal activity."
        )
    
    # Insight 5: Seasonal patterns
    if 'Month_Name' in df.columns:
        peak_month = df['Month_Name'].value_counts().idxmax()
        low_month = df['Month_Name'].value_counts().idxmin()
        insights.append(
            f"5. SEASONAL VARIATION: Crime is highest in {peak_month} and lowest in "
            f"{low_month}, consistent with weather-related patterns."
        )
    
    # Insight 6: Location
    if loc_desc_col and loc_desc_col in df.columns:
        top_loc = df[loc_desc_col].value_counts().head(1)
        top_loc_pct = top_loc.values[0] / len(df) * 100
        insights.append(
            f"6. LOCATION CONCENTRATION: '{top_loc.index[0]}' is the most common "
            f"crime location ({top_loc_pct:.1f}% of all incidents)."
        )
    
    # Insight 7: Domestic
    if dom_col and dom_col in df.columns:
        insights.append(
            f"7. DOMESTIC CRIMES: {domestic_rate:.1f}% of all crimes are classified "
            f"as domestic, highlighting a substantial domestic crime component."
        )
    
    # Insight 8: Data quality
    if len(missing_df) > 0:
        high_missing = missing_df[missing_df['Missing_Percentage'] > 5]
        if len(high_missing) > 0:
            cols_list = ', '.join(high_missing.index.tolist()[:3])
            insights.append(
                f"8. DATA QUALITY: Columns with significant missing data include "
                f"{cols_list}. These should be accounted for in any downstream analysis."
            )
        else:
            insights.append(
                f"8. DATA QUALITY: Missing values are minimal across the dataset, "
                f"indicating generally good data quality."
            )
    
    for insight in insights:
        print(insight)
        print()
    
    # ========================================================================
    # 14. EXECUTIVE SUMMARY
    # ========================================================================
    separator("14. EXECUTIVE SUMMARY")
    
    print("CHICAGO CRIME DATASET — EXECUTIVE SUMMARY")
    print("-" * 50)
    print(f"• Dataset Size: {len(df):,} records, {df.shape[1]} columns")
    if date_col_actual and date_col_actual in df.columns:
        print(f"• Time Period: {df[date_col_actual].min()} to {df[date_col_actual].max()}")
    if pt_col and pt_col in df.columns:
        print(f"• Crime Types: {df[pt_col].nunique()} unique categories")
        top3 = df[pt_col].value_counts().head(3)
        print(f"• Top 3 Crimes: {', '.join(top3.index)}")
    if arr_col and arr_col in df.columns:
        print(f"• Overall Arrest Rate: {overall_arrest_rate:.1f}%")
    if 'Hour' in df.columns:
        print(f"• Peak Crime Hour: {df['Hour'].value_counts().idxmax()}:00")
    if 'Month_Name' in df.columns:
        print(f"• Peak Crime Month: {df['Month_Name'].value_counts().idxmax()}")
    if len(missing_df) > 0:
        max_miss = missing_df['Missing_Percentage'].max()
        print(f"• Max Missing Data: {max_miss:.1f}% ({missing_df.index[0]})")
    print(f"• Duplicated Rows: {n_dupes:,}")
    
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {os.path.abspath(PLOT_DIR)}/")


if __name__ == '__main__':
    main()

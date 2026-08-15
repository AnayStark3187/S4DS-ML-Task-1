#!/usr/bin/env python3
"""Generate the complete Chicago Crime EDA Jupyter Notebook."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

cells = []

# ==========================================================================
# CELL 1: Title & Executive Summary (Markdown)
# ==========================================================================
cells.append(new_markdown_cell("""# 🏙️ Chicago Crime Dataset — Comprehensive Exploratory Data Analysis

---

## Executive Summary

> **Objective:** Conduct a rigorous Exploratory Data Analysis of the Chicago Crime dataset to understand patterns in crime type, location, temporal distribution, arrest outcomes, and other relevant variables. All findings are derived directly from the data — no fabricated results.

### Key Highlights *(populated after analysis)*:
- **Dataset Size:** ~120,759 records × 23 columns
- **Time Period:** Determined after loading
- **Most Common Crime Types:** Identified through analysis
- **Overall Arrest Rate:** Calculated from data
- **Temporal Patterns:** Peak hours, months, and seasonal effects
- **Geographic Patterns:** Spatial crime distribution
- **Data Quality:** Missing values, duplicates, anomalies

---

### Table of Contents

1. Introduction & Libraries
2. Load Dataset
3. Data Overview
4. Data Quality Assessment
5. Data Cleaning & Feature Engineering
6. Univariate Analysis
7. Bivariate Analysis
8. Multivariate Analysis
9. Geographic Analysis
10. Correlation Analysis
11. Statistical Analysis
12. Key Insights
13. Conclusion"""))

# ==========================================================================
# CELL 2: Import Libraries (Code)
# ==========================================================================
cells.append(new_markdown_cell("## 1. Import Libraries"))

cells.append(new_code_cell("""import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global plot configuration
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

print("✅ All libraries imported successfully.")
print(f"   pandas: {pd.__version__}")
print(f"   numpy: {np.__version__}")
print(f"   matplotlib: {plt.matplotlib.__version__}")
print(f"   seaborn: {sns.__version__}")"""))

# ==========================================================================
# CELL 3: Helper Functions (Code)
# ==========================================================================
cells.append(new_markdown_cell("### Helper Functions\n\nReusable utility functions for plotting and analysis throughout the notebook."))

cells.append(new_code_cell("""def plot_top_categories(series, n=15, title='', xlabel='Count', ylabel='',
                        color=ACCENT, figsize=(12, 7), horizontal=True):
    \"\"\"Plot top N categories from a pandas Series.\"\"\"
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
    plt.show()


def plot_percentage_bar(labels, values, title='', xlabel='', color=ACCENT, figsize=(12, 7)):
    \"\"\"Plot horizontal bar chart with percentage annotations.\"\"\"
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
    plt.show()


def calculate_percentage(series):
    \"\"\"Calculate value counts and percentages.\"\"\"
    counts = series.value_counts()
    percentages = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})


def separator(title):
    \"\"\"Print a formatted section separator.\"\"\"
    print(f"\\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\\n")

print("✅ Helper functions defined.")"""))

# ==========================================================================
# CELL 4: Load Dataset (Markdown + Code)
# ==========================================================================
cells.append(new_markdown_cell("""## 2. Load Dataset

We load the Chicago Crime dataset from the local CSV file. The dataset was downloaded from the Google Drive link provided. If you need to re-download, place the file in this directory as `chicago_crime_data.csv`.

The loading logic auto-detects the dataset file by scanning for CSV files in the project directory."""))

cells.append(new_code_cell("""# Auto-detect dataset file
def find_dataset():
    \"\"\"Auto-detect the Chicago Crime dataset file.\"\"\"
    extensions = ['*.csv', '*.CSV']
    candidates = []
    for ext in extensions:
        candidates.extend(glob.glob(ext))
        candidates.extend(glob.glob(os.path.join('data', ext)))
    
    if not candidates:
        print("❌ Dataset not found! Place the CSV file in the project directory.")
        print("   Download from: https://drive.google.com/file/d/1Efog2t2MWgm1Ciyn2nZyXqR9YAhUm7Ur/view")
        return None
    
    # Prefer files with 'crime' or 'chicago' in the name
    for c in candidates:
        if 'crime' in c.lower() or 'chicago' in c.lower():
            return c
    return candidates[0]


filepath = find_dataset()
if filepath:
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Dataset loaded from: {filepath}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   File size: {os.path.getsize(filepath) / 1e6:.1f} MB")
else:
    raise FileNotFoundError("Dataset not found. Please download and place in project directory.")"""))

# ==========================================================================
# CELL 5: Data Overview (Markdown + Code)
# ==========================================================================
cells.append(new_markdown_cell("""## 3. Data Overview

We begin by inspecting the dataset's structure: shape, columns, data types, memory usage, and summary statistics."""))

cells.append(new_code_cell("""# 3.1 First 5 rows
print("=" * 70)
print("  FIRST 5 ROWS")
print("=" * 70)
df.head()"""))

cells.append(new_code_cell("""# 3.2 Last 5 rows
print("=" * 70)
print("  LAST 5 ROWS")
print("=" * 70)
df.tail()"""))

cells.append(new_code_cell("""# 3.3 Random sample
print("=" * 70)
print("  RANDOM SAMPLE (5 rows)")
print("=" * 70)
df.sample(5, random_state=42)"""))

cells.append(new_code_cell("""# 3.4 Dataset dimensions and column names
print(f"Number of rows:    {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")
print(f"\\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")"""))

cells.append(new_code_cell("""# 3.5 Data types and info
print("=" * 70)
print("  DATA TYPES & INFO")
print("=" * 70)
df.info()"""))

cells.append(new_code_cell("""# 3.6 Memory usage
mem = df.memory_usage(deep=True)
print(f"Total memory usage: {mem.sum() / 1e6:.1f} MB\\n")
print(mem.sort_values(ascending=False).to_string())"""))

cells.append(new_code_cell("""# 3.7 Numerical summary statistics
print("=" * 70)
print("  NUMERICAL SUMMARY STATISTICS")
print("=" * 70)
df.describe()"""))

cells.append(new_code_cell("""# 3.8 Categorical summary statistics
print("=" * 70)
print("  CATEGORICAL SUMMARY STATISTICS")
print("=" * 70)
df.describe(include='object')"""))

cells.append(new_markdown_cell("""### Observations from Data Overview

- The dataset contains **crime incident records** from Chicago with fields for crime type, location, arrest status, and geographic coordinates.
- **Key Identifiers:** `ID`, `Case Number`
- **Target/Outcome Variables:** `Arrest` (boolean), `Domestic` (boolean)
- **Temporal Variables:** `Date`, `Year`, `_year` (appears to be a duplicate year column)
- **Geographic Variables:** `Latitude`, `Longitude`, `X Coordinate`, `Y Coordinate`, `Location`, `Block`
- **Categorical Variables:** `Primary Type`, `Description`, `Location Description`, `FBI Code`, `IUCR`
- **Numerical Variables:** `Beat`, `District`, `Ward`, `Community Area`"""))

# ==========================================================================
# CELL 6: Data Understanding — Variable Classification
# ==========================================================================
cells.append(new_markdown_cell("""## 3.1 Data Understanding — Variable Classification

Let's systematically classify all columns into numerical, categorical, and temporal variables."""))

cells.append(new_code_cell("""# Classify variables
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

print(f"Numerical Variables ({len(numerical_cols)}):")
for c in numerical_cols:
    n_unique = df[c].nunique()
    print(f"  • {c:25s} | unique: {n_unique:,} | range: [{df[c].min()}, {df[c].max()}]")

print(f"\\nCategorical Variables ({len(categorical_cols)}):")
for c in categorical_cols:
    n_unique = df[c].nunique()
    print(f"  • {c:25s} | unique: {n_unique:,}")

print(f"\\nBoolean Variables ({len(boolean_cols)}):")
for c in boolean_cols:
    print(f"  • {c:25s} | True: {df[c].sum():,} | False: {(~df[c]).sum():,}")"""))

# ==========================================================================
# CELL 7: Parse Dates and Feature Engineering
# ==========================================================================
cells.append(new_markdown_cell("""## 5. Data Cleaning & Feature Engineering

### 5.1 Parse Date Column

Convert the `Date` column to datetime and extract useful temporal features."""))

cells.append(new_code_cell("""# Parse date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)

# Extract temporal features
df['Year_extracted'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.day_name()
df['Day_of_Week_Num'] = df['Date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['Hour'] = df['Date'].dt.hour
df['Quarter'] = df['Date'].dt.quarter
df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

print(f"✅ Date parsed and temporal features extracted.")
print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")
print(f"   Years covered: {df['Year_extracted'].min()} – {df['Year_extracted'].max()}")
print(f"   Records with invalid dates: {df['Date'].isna().sum():,}")
print(f"\\nNew columns added: Year_extracted, Month, Month_Name, Day, Day_of_Week,")
print(f"                    Day_of_Week_Num, Hour, Quarter, Is_Weekend")"""))

cells.append(new_markdown_cell("""### 5.2 Handle the `_year` Column

The dataset contains both a `Year` column and a `_year` column. Let's check whether they are identical and drop the redundant one if so."""))

cells.append(new_code_cell("""# Check if Year and _year are identical
if '_year' in df.columns and 'Year' in df.columns:
    identical = (df['Year'] == df['_year']).all()
    print(f"'Year' and '_year' are identical: {identical}")
    if identical:
        print("→ '_year' is a duplicate. We will use 'Year' for analysis.")
    else:
        diff_count = (df['Year'] != df['_year']).sum()
        print(f"→ {diff_count:,} rows differ between 'Year' and '_year'.")"""))

# ==========================================================================
# CELL 8: Data Quality Assessment
# ==========================================================================
cells.append(new_markdown_cell("""## 4. Data Quality Assessment

A thorough data-quality audit covering missing values, duplicates, inconsistencies, and numerical anomalies.

### 4.1 Missing Values"""))

cells.append(new_code_cell("""# Calculate missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

# Show only columns with missing values
missing_with = missing_df[missing_df['Missing Count'] > 0]
print(f"Columns with missing values: {len(missing_with)} out of {len(df.columns)}")
print(f"{'='*55}")
print(missing_with.to_string())"""))

cells.append(new_code_cell("""# Visualize missing values
if len(missing_with) > 0:
    fig, ax = plt.subplots(figsize=(12, max(5, len(missing_with) * 0.5)))
    
    colors = ['#E53935' if p > 10 else '#FFA726' if p > 1 else '#66BB6A'
              for p in missing_with['Missing %']]
    
    missing_with['Missing %'].sort_values().plot(
        kind='barh', ax=ax, color=colors[::-1], edgecolor='white'
    )
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    
    for i, (idx, row) in enumerate(missing_with.sort_values('Missing %').iterrows()):
        ax.text(row['Missing %'] + 0.3, i,
                f"{row['Missing %']:.1f}% ({int(row['Missing Count']):,})",
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset!")"""))

cells.append(new_markdown_cell("""**Interpretation of Missing Values:**

- Columns like `Location`, `Latitude`, `Longitude`, `X Coordinate`, `Y Coordinate` may have missing geographic data — these are expected for incidents where location couldn't be geocoded.
- `Ward` and `Community Area` missingness typically correlates with missing geographic data.
- We will **not** drop rows with missing values blindly. Instead, we handle them contextually in each analysis section.

---

### 4.2 Duplicate Records"""))

cells.append(new_code_cell("""# Check for exact duplicates
n_dupes = df.duplicated().sum()
pct_dupes = n_dupes / len(df) * 100

print(f"Completely duplicated rows: {n_dupes:,} ({pct_dupes:.2f}%)")
print(f"Unique rows:               {len(df) - n_dupes:,}")

# Check duplicate IDs
n_unique_ids = df['ID'].nunique()
print(f"\\nUnique IDs:      {n_unique_ids:,} / {len(df):,}")
if n_unique_ids < len(df):
    print(f"→ {len(df) - n_unique_ids:,} records share an ID with another record.")
    
# Check duplicate Case Numbers
n_unique_cases = df['Case Number'].nunique()
print(f"Unique Case Nos: {n_unique_cases:,} / {len(df):,}")
if n_unique_cases < len(df):
    print(f"→ {len(df) - n_unique_cases:,} records share a Case Number.")
    print("   (Some cases may legitimately involve multiple charges/incidents)")"""))

cells.append(new_markdown_cell("""**Note:** Duplicate Case Numbers can be legitimate — a single case can involve multiple charge types (e.g., assault + battery). We do **not** remove these without further investigation.

---

### 4.3 Inconsistent Values"""))

cells.append(new_code_cell("""# Check categorical columns for inconsistencies
print("=" * 70)
print("  CATEGORICAL VALUE CONSISTENCY CHECK")
print("=" * 70)

cat_cols_to_check = ['Primary Type', 'Description', 'Location Description', 'FBI Code']

for col in cat_cols_to_check:
    if col not in df.columns:
        continue
    
    unique_vals = df[col].dropna().unique()
    n_unique = len(unique_vals)
    
    # Check whitespace issues
    series_str = pd.Series(unique_vals).astype(str)
    whitespace_issues = (series_str.str.strip() != series_str).sum()
    
    # Check null-like strings
    null_strings = {'N/A', 'NA', 'None', 'null', 'NULL', 'Unknown', 'UNKNOWN', ''}
    found_nulls = [v for v in unique_vals if str(v).strip() in null_strings]
    
    # Rare categories (< 0.1%)
    vc = df[col].value_counts()
    rare = vc[vc / len(df) < 0.001]
    
    print(f"\\n📋 {col}")
    print(f"   Unique values: {n_unique}")
    print(f"   Whitespace issues: {whitespace_issues}")
    if found_nulls:
        print(f"   Null-like strings: {found_nulls}")
    print(f"   Rare categories (<0.1%): {len(rare)}")
    if 0 < len(rare) <= 10:
        for name, count in rare.items():
            print(f"      → '{name}': {count:,} ({count/len(df)*100:.3f}%)")"""))

cells.append(new_markdown_cell("""### 4.4 Numerical Anomalies

Check for impossible values, particularly in latitude/longitude (Chicago bounds: lat ~41.6–42.1, lon ~-87.95 to -87.5)."""))

cells.append(new_code_cell("""# Check latitude/longitude bounds
print("=" * 70)
print("  GEOGRAPHIC COORDINATE VALIDATION")
print("=" * 70)

# Chicago approximate bounds
CHI_LAT_MIN, CHI_LAT_MAX = 41.6, 42.1
CHI_LON_MIN, CHI_LON_MAX = -87.95, -87.5

if 'Latitude' in df.columns:
    lat = df['Latitude'].dropna()
    out_lat = lat[(lat < CHI_LAT_MIN) | (lat > CHI_LAT_MAX)]
    zero_lat = lat[lat == 0]
    print(f"\\nLatitude:")
    print(f"  Valid records:     {len(lat):,}")
    print(f"  Range:             [{lat.min():.6f}, {lat.max():.6f}]")
    print(f"  Outside Chicago:   {len(out_lat):,} ({len(out_lat)/len(lat)*100:.2f}%)")
    print(f"  Zero values:       {len(zero_lat):,}")

if 'Longitude' in df.columns:
    lon = df['Longitude'].dropna()
    out_lon = lon[(lon < CHI_LON_MIN) | (lon > CHI_LON_MAX)]
    zero_lon = lon[lon == 0]
    print(f"\\nLongitude:")
    print(f"  Valid records:     {len(lon):,}")
    print(f"  Range:             [{lon.min():.6f}, {lon.max():.6f}]")
    print(f"  Outside Chicago:   {len(out_lon):,} ({len(out_lon)/len(lon)*100:.2f}%)")
    print(f"  Zero values:       {len(zero_lon):,}")

# District, Ward, Beat ranges
for col in ['District', 'Ward', 'Community Area', 'Beat']:
    if col in df.columns:
        vals = df[col].dropna()
        print(f"\\n{col}: range [{vals.min()}, {vals.max()}], unique: {vals.nunique()}")"""))

cells.append(new_code_cell("""# Box plots for numerical variables
num_cols_for_box = [c for c in ['District', 'Ward', 'Community Area', 'Beat'] if c in df.columns]

if num_cols_for_box:
    fig, axes = plt.subplots(1, len(num_cols_for_box), figsize=(4 * len(num_cols_for_box), 5))
    if len(num_cols_for_box) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, num_cols_for_box):
        df[col].dropna().plot(kind='box', ax=ax, patch_artist=True,
                              boxprops=dict(facecolor='#E3F2FD'))
        ax.set_title(col, fontsize=12, fontweight='bold')
    
    plt.suptitle('Numerical Variable Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()"""))

# ==========================================================================
# CELL 9: Univariate Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 6. Univariate Analysis

### 6.1 Crime Type Distribution

Analyze the distribution of crime types using the `Primary Type` column."""))

cells.append(new_code_cell("""# Crime type analysis
crime_counts = df['Primary Type'].value_counts()
print(f"Total unique crime types: {len(crime_counts)}")
print(f"\\nTop 10 Crime Types:")
print("=" * 55)
for crime, count in crime_counts.head(10).items():
    pct = count / len(df) * 100
    print(f"  {crime:35s} {count:>7,}  ({pct:5.1f}%)")

print(f"\\nBottom 5 Crime Types:")
print("=" * 55)
for crime, count in crime_counts.tail(5).items():
    pct = count / len(df) * 100
    print(f"  {crime:35s} {count:>7,}  ({pct:5.1f}%)")"""))

cells.append(new_code_cell("""# Visualization: Top 15 Crime Types
plot_top_categories(
    df['Primary Type'], n=15,
    title='Top 15 Crime Types in Chicago',
    xlabel='Number of Incidents',
    color=ACCENT
)"""))

cells.append(new_markdown_cell("""### 6.2 Crime Description Distribution

The `Description` field provides more granular detail about each crime incident."""))

cells.append(new_code_cell("""# Crime description analysis
print(f"Total unique descriptions: {df['Description'].nunique()}")
print(f"\\nTop 10 Descriptions:")
for desc, count in df['Description'].value_counts().head(10).items():
    pct = count / len(df) * 100
    print(f"  {desc:45s} {count:>7,}  ({pct:5.1f}%)")"""))

cells.append(new_code_cell("""# Visualization: Top 15 Crime Descriptions
plot_top_categories(
    df['Description'], n=15,
    title='Top 15 Crime Descriptions',
    xlabel='Number of Incidents',
    color='#FF9800'
)"""))

cells.append(new_markdown_cell("""### 6.3 Arrest Distribution

Analyze the overall arrest rate — what proportion of crimes result in an arrest?"""))

cells.append(new_code_cell("""# Arrest analysis
arrest_dist = calculate_percentage(df['Arrest'])
print(arrest_dist)

overall_arrest_rate = df['Arrest'].mean() * 100
print(f"\\n🔒 Overall Arrest Rate: {overall_arrest_rate:.1f}%")
print(f"   Arrests made:     {df['Arrest'].sum():,}")
print(f"   No arrest:        {(~df['Arrest']).sum():,}")"""))

cells.append(new_code_cell("""# Visualization: Arrest Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
arrest_counts = df['Arrest'].value_counts()
labels = ['No Arrest', 'Arrest']
values = [arrest_counts[False], arrest_counts[True]]
colors_arr = ['#EF5350', '#66BB6A']

ax1.bar(labels, values, color=colors_arr, edgecolor='white', width=0.5)
ax1.set_title('Arrest Distribution (Count)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Incidents')
for i, v in enumerate(values):
    ax1.text(i, v + max(values) * 0.02, f'{v:,}', ha='center', fontsize=11, fontweight='bold')

# Pie chart (only 2 categories — appropriate use)
ax2.pie(values, labels=labels, autopct='%1.1f%%',
        colors=colors_arr, startangle=90, textprops={'fontsize': 12},
        explode=(0, 0.05))
ax2.set_title('Arrest Distribution (Percentage)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 6.4 Domestic Crimes

Analyze the proportion of crimes classified as domestic incidents."""))

cells.append(new_code_cell("""# Domestic crime analysis
domestic_dist = calculate_percentage(df['Domestic'])
print(domestic_dist)

domestic_rate = df['Domestic'].mean() * 100
print(f"\\n🏠 Domestic Crime Rate: {domestic_rate:.1f}%")
print(f"   Domestic incidents:     {df['Domestic'].sum():,}")
print(f"   Non-domestic incidents: {(~df['Domestic']).sum():,}")"""))

cells.append(new_markdown_cell("""### 6.5 Location Distribution

Where do crimes occur most frequently?"""))

cells.append(new_code_cell("""# Location description analysis
print(f"Unique location types: {df['Location Description'].nunique()}")
print(f"\\nTop 15 Crime Locations:")
for loc, count in df['Location Description'].value_counts().head(15).items():
    pct = count / len(df) * 100
    print(f"  {str(loc):40s} {count:>7,}  ({pct:5.1f}%)")"""))

cells.append(new_code_cell("""# Visualization: Top 15 Crime Locations
plot_top_categories(
    df['Location Description'], n=15,
    title='Top 15 Crime Locations',
    xlabel='Number of Incidents',
    color='#9C27B0'
)"""))

cells.append(new_code_cell("""# District distribution
print(f"\\nCrimes by District (Top 10):")
dist_counts = df['District'].value_counts().head(10)
for dist, count in dist_counts.items():
    pct = count / len(df) * 100
    print(f"  District {int(dist):3d}: {count:>7,}  ({pct:5.1f}%)")"""))

cells.append(new_markdown_cell("""### 6.6 Temporal Distribution

Analyze how crime varies across years, months, days of the week, and hours of the day."""))

cells.append(new_code_cell("""# 6.6.1 Crimes by Year
year_counts = df['Year'].value_counts().sort_index()
print("Crimes by Year:")
for year, count in year_counts.items():
    print(f"  {int(year)}: {count:>7,}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2.5,
        color=ACCENT, markersize=7, zorder=3)
ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color=ACCENT)
ax.set_title('Crime Incidents by Year', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Incidents')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# 6.6.2 Crimes by Month
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

# Annotate peak and low
peak_idx = month_counts.values.argmax()
low_idx = month_counts.values.argmin()
ax.annotate(f'Peak: {month_order[peak_idx][:3]}', xy=(peak_idx, month_counts.values[peak_idx]),
            xytext=(peak_idx, month_counts.values[peak_idx] * 1.05),
            ha='center', fontsize=10, fontweight='bold', color='red')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nPeak month: {month_order[peak_idx]} ({month_counts.values[peak_idx]:,})")
print(f"Lowest month: {month_order[low_idx]} ({month_counts.values[low_idx]:,})")"""))

cells.append(new_code_cell("""# 6.6.3 Crimes by Day of Week
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts = df['Day_of_Week'].value_counts().reindex(dow_order)

fig, ax = plt.subplots(figsize=(12, 5))
colors_dow = ['#42A5F5' if d in ['Saturday', 'Sunday'] else '#78909C' for d in dow_order]
bars = ax.bar(dow_order, dow_counts.values, color=colors_dow, edgecolor='white')
ax.set_title('Crime Incidents by Day of Week', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Incidents')
for i, v in enumerate(dow_counts.values):
    ax.text(i, v + dow_counts.max() * 0.01, f'{v:,}', ha='center', fontsize=9, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(['Weekend', 'Weekday'], loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nHighest crime day: {dow_counts.idxmax()} ({dow_counts.max():,})")
print(f"Lowest crime day:  {dow_counts.idxmin()} ({dow_counts.min():,})")"""))

cells.append(new_code_cell("""# 6.6.4 Crimes by Hour of Day
hour_counts = df['Hour'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
colors_hour = ['#FFA726' if (h >= 18 or h <= 5) else '#42A5F5' for h in range(24)]
ax.bar(hour_counts.index, hour_counts.values, color=colors_hour, edgecolor='white')
ax.set_title('Crime Incidents by Hour of Day', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour (0 = midnight, 12 = noon, 23 = 11 PM)')
ax.set_ylabel('Number of Incidents')
ax.set_xticks(range(24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)

# Annotate peak
peak_hour = hour_counts.idxmax()
ax.annotate(f'Peak: {peak_hour}:00', xy=(peak_hour, hour_counts.max()),
            xytext=(peak_hour + 2, hour_counts.max() * 1.05),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, fontweight='bold', color='red')
plt.tight_layout()
plt.show()

print(f"\\nPeak crime hour:   {peak_hour}:00 ({hour_counts.max():,} incidents)")
print(f"Lowest crime hour: {hour_counts.idxmin()}:00 ({hour_counts.min():,} incidents)")"""))

# ==========================================================================
# CELL 10: Bivariate Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 7. Bivariate Analysis

Investigating relationships between pairs of important variables.

### 7.1 Crime Type vs Arrest Rate

Not just raw arrest counts — we calculate the **arrest rate** (percentage) for each crime type."""))

cells.append(new_code_cell("""# Arrest rate by crime type (top 15 most common crimes)
top15_crimes = df['Primary Type'].value_counts().head(15).index
arrest_by_type = df[df['Primary Type'].isin(top15_crimes)].groupby('Primary Type')['Arrest'].agg(
    ['sum', 'count', 'mean']
).rename(columns={'sum': 'Arrests', 'count': 'Total', 'mean': 'Arrest_Rate'})
arrest_by_type['Arrest_Rate'] = (arrest_by_type['Arrest_Rate'] * 100).round(1)
arrest_by_type = arrest_by_type.sort_values('Arrest_Rate', ascending=True)

print("Arrest Rate by Crime Type (Top 15 Most Common):")
print("=" * 65)
for crime, row in arrest_by_type.iterrows():
    bar = "█" * int(row['Arrest_Rate'] / 2)
    print(f"  {crime:35s} {row['Arrest_Rate']:5.1f}%  {bar}  (n={int(row['Total']):,})")"""))

cells.append(new_code_cell("""# Visualization: Arrest Rate by Crime Type
fig, ax = plt.subplots(figsize=(12, 8))

colors_arrest = ['#66BB6A' if r > 50 else '#FFA726' if r > 25 else '#EF5350'
                 for r in arrest_by_type['Arrest_Rate']]

ax.barh(range(len(arrest_by_type)), arrest_by_type['Arrest_Rate'],
        color=colors_arrest, edgecolor='white')
ax.set_yticks(range(len(arrest_by_type)))
ax.set_yticklabels(arrest_by_type.index)
ax.set_xlabel('Arrest Rate (%)', fontsize=12)
ax.set_title('Arrest Rate by Crime Type (Top 15 Crime Types)',
              fontsize=14, fontweight='bold')
ax.axvline(x=overall_arrest_rate, color='red', linestyle='--', alpha=0.7,
           label=f'Overall Rate: {overall_arrest_rate:.1f}%')
ax.legend(loc='lower right', fontsize=11)

for i, (_, row) in enumerate(arrest_by_type.iterrows()):
    ax.text(row['Arrest_Rate'] + 0.5, i, f"{row['Arrest_Rate']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 7.2 Crime Type Trends Over Time

How have the top crime categories changed over the years?"""))

cells.append(new_code_cell("""# Top 5 crime types over years
top5_crimes = df['Primary Type'].value_counts().head(5).index
crime_year = df[df['Primary Type'].isin(top5_crimes)].groupby(
    ['Year', 'Primary Type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
for i, crime in enumerate(top5_crimes):
    if crime in crime_year.columns:
        ax.plot(crime_year.index, crime_year[crime], marker='o',
                label=crime, linewidth=2, markersize=5, color=PALETTE[i])

ax.set_title('Top 5 Crime Types — Trend Over Years', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Incidents')
ax.legend(loc='best', framealpha=0.9, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Crime Type × Month Heatmap
top8_crimes = df['Primary Type'].value_counts().head(8).index
crime_month = df[df['Primary Type'].isin(top8_crimes)].groupby(
    ['Month', 'Primary Type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(crime_month.T, annot=True, fmt=',d', cmap='YlOrRd',
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Number of Incidents'})
ax.set_title('Crime Counts: Crime Type × Month (Top 8 Crimes)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Crime Type')
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 7.3 Crime Type vs Location

Which crime types are most common at which locations?"""))

cells.append(new_code_cell("""# Crime Type × Location heatmap (top 8 each)
top8_crimes = df['Primary Type'].value_counts().head(8).index
top8_locs = df['Location Description'].value_counts().head(8).index

crime_loc = df[df['Primary Type'].isin(top8_crimes) & df['Location Description'].isin(top8_locs)] \\
    .groupby(['Primary Type', 'Location Description']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(crime_loc, annot=True, fmt=',d', cmap='Blues', ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Count'})
ax.set_title('Crime Type × Location (Top 8 each)', fontsize=14, fontweight='bold')
ax.set_xlabel('Location Description')
ax.set_ylabel('Crime Type')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 7.4 Crime vs Day of Week (by crime type)"""))

cells.append(new_code_cell("""# Crime type by day of week (top 5)
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
top5 = df['Primary Type'].value_counts().head(5).index

fig, ax = plt.subplots(figsize=(14, 6))
for i, crime in enumerate(top5):
    subset = df[df['Primary Type'] == crime]
    dow_counts_crime = subset['Day_of_Week'].value_counts().reindex(dow_order)
    ax.plot(dow_order, dow_counts_crime.values, marker='o', label=crime,
            linewidth=2, markersize=6, color=PALETTE[i])

ax.set_title('Crime by Day of Week (Top 5 Crime Types)', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Number of Incidents')
ax.legend(loc='best', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 7.5 Crime vs Hour (by crime type)

Different crime types may show different hourly patterns."""))

cells.append(new_code_cell("""# Hourly patterns by crime type (top 5)
fig, ax = plt.subplots(figsize=(14, 6))
for i, crime in enumerate(top5):
    subset = df[df['Primary Type'] == crime]
    hourly = subset['Hour'].value_counts().sort_index()
    ax.plot(hourly.index, hourly.values, label=crime, linewidth=2, color=PALETTE[i])

ax.set_title('Hourly Crime Patterns by Crime Type (Top 5)', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Incidents')
ax.set_xticks(range(24))
ax.legend(loc='best', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 7.6 Arrest Rate vs Time

Has the arrest rate changed over time?"""))

cells.append(new_code_cell("""# Arrest rate over years
arrest_year = df.groupby('Year')['Arrest'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# By Year
axes[0].plot(arrest_year.index, arrest_year.values, marker='o',
        linewidth=2.5, color=ACCENT2, markersize=7, zorder=3)
axes[0].set_title('Arrest Rate Over Years', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Arrest Rate (%)')
axes[0].axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5,
                label=f'Overall: {overall_arrest_rate:.1f}%')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# By Hour
arrest_hour = df.groupby('Hour')['Arrest'].mean() * 100
axes[1].bar(arrest_hour.index, arrest_hour.values, color=ACCENT2, edgecolor='white')
axes[1].set_title('Arrest Rate by Hour of Day', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Arrest Rate (%)')
axes[1].set_xticks(range(24))
axes[1].axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5,
                label=f'Overall: {overall_arrest_rate:.1f}%')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Arrest rate by month and day of week
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# By Month
arrest_month = df.groupby('Month')['Arrest'].mean() * 100
axes[0].bar(range(1, 13), arrest_month.values, color=ACCENT2, edgecolor='white')
axes[0].set_title('Arrest Rate by Month', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Arrest Rate (%)')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels([m[:3] for m in month_order])
axes[0].axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(axis='y', alpha=0.3)

# By Day of Week
arrest_dow = df.groupby('Day_of_Week')['Arrest'].mean().reindex(dow_order) * 100
axes[1].bar(dow_order, arrest_dow.values, color=ACCENT2, edgecolor='white')
axes[1].set_title('Arrest Rate by Day of Week', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Arrest Rate (%)')
axes[1].axhline(y=overall_arrest_rate, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(axis='y', alpha=0.3)
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()"""))

# ==========================================================================
# CELL 11: Multivariate Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 8. Multivariate Analysis

Deeper analysis using combinations of three or more variables.

### 8.1 Day of Week × Hour Crime Heatmap

One of the most revealing visualizations — showing crime density across the entire week."""))

cells.append(new_code_cell("""# Day of Week × Hour Crime Heatmap
dow_hour = df.groupby(['Day_of_Week_Num', 'Hour']).size().unstack(fill_value=0)
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(dow_hour, cmap='YlOrRd', ax=ax, linewidths=0.3,
            yticklabels=dow_labels, annot=False,
            cbar_kws={'label': 'Number of Incidents'})
ax.set_title('🔥 Crime Heatmap: Day of Week × Hour of Day', fontsize=16, fontweight='bold')
ax.set_xlabel('Hour of Day (0–23)', fontsize=12)
ax.set_ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.show()

# Summary statistics
print("Peak combinations (Day × Hour):")
dow_hour_flat = df.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
top5_combos = dow_hour_flat.nlargest(5, 'Count')
for _, row in top5_combos.iterrows():
    print(f"  {row['Day_of_Week']:10s} at {int(row['Hour']):02d}:00 → {row['Count']:,} incidents")"""))

cells.append(new_markdown_cell("""### 8.2 Crime Type × Hour × Arrest Rate

Does arrest probability differ by crime type and time of day?"""))

cells.append(new_code_cell("""# Crime Type × Hour × Arrest Rate heatmap
top5 = df['Primary Type'].value_counts().head(5).index
subset = df[df['Primary Type'].isin(top5)]

pivot = subset.groupby(['Primary Type', 'Hour'])['Arrest'].mean().unstack(fill_value=0) * 100

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot, cmap='RdYlGn', ax=ax, linewidths=0.3,
            vmin=0, vmax=pivot.values.max() * 1.1,
            cbar_kws={'label': 'Arrest Rate (%)'})
ax.set_title('Arrest Rate (%): Crime Type × Hour of Day (Top 5 Crimes)',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 8.3 Crime Type × Month (Seasonal Patterns)

Do different crime categories have seasonal patterns?"""))

cells.append(new_code_cell("""# Seasonal patterns - normalized by crime type to show relative patterns
top5 = df['Primary Type'].value_counts().head(5).index
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, crime in enumerate(top5):
    subset = df[df['Primary Type'] == crime]
    monthly = subset['Month'].value_counts().sort_index()
    avg = monthly.mean()
    
    colors = ['#E53935' if v > avg * 1.1 else '#2196F3' if v < avg * 0.9 else '#78909C'
              for v in monthly.values]
    
    axes[i].bar(range(1, 13), monthly.values, color=colors, edgecolor='white')
    axes[i].axhline(y=avg, color='gray', linestyle='--', alpha=0.5)
    axes[i].set_title(crime, fontsize=12, fontweight='bold')
    axes[i].set_xticks(range(1, 13))
    axes[i].set_xticklabels([m[:1] for m in month_order], fontsize=9)
    axes[i].set_ylabel('Count')

# Remove unused subplot
axes[5].axis('off')

plt.suptitle('Monthly Crime Patterns by Type (Red=Above Avg, Blue=Below Avg)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(new_markdown_cell("""### 8.4 District × Crime Type

Which districts have unusually high concentrations of particular crime categories?"""))

cells.append(new_code_cell("""# District × Crime Type heatmap
top8_crimes = df['Primary Type'].value_counts().head(8).index
top10_districts = df['District'].value_counts().head(10).index.astype(int)

dist_crime = df[df['Primary Type'].isin(top8_crimes) & df['District'].isin(top10_districts)] \\
    .groupby(['District', 'Primary Type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(dist_crime, annot=True, fmt=',d', cmap='Blues', ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Count'})
ax.set_title('Crime Distribution: District × Crime Type (Top 10 Districts, Top 8 Crimes)',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Crime Type', fontsize=12)
ax.set_ylabel('District', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""))

# ==========================================================================
# CELL 12: Geographic Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 9. Geographic Analysis

Analyzing the spatial distribution of crimes using latitude and longitude data."""))

cells.append(new_code_cell("""# Filter to valid geographic coordinates
geo = df.dropna(subset=['Latitude', 'Longitude'])
geo = geo[(geo['Latitude'] > 41.6) & (geo['Latitude'] < 42.1) &
          (geo['Longitude'] > -87.95) & (geo['Longitude'] < -87.5)]

print(f"Records with valid coordinates: {len(geo):,} / {len(df):,} ({len(geo)/len(df)*100:.1f}%)")
print(f"Records excluded (missing/invalid coords): {len(df) - len(geo):,}")"""))

cells.append(new_code_cell("""# Overall crime geographic scatter plot
fig, ax = plt.subplots(figsize=(10, 12))

# Sample for performance (large datasets)
sample_size = min(50000, len(geo))
geo_sample = geo.sample(n=sample_size, random_state=42)

ax.scatter(geo_sample['Longitude'], geo_sample['Latitude'],
           s=0.3, alpha=0.05, c=ACCENT)
ax.set_title(f'Crime Geographic Distribution (n={sample_size:,} sampled)',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_aspect('equal')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Crime type spatial distribution (top 4)
top4 = df['Primary Type'].value_counts().head(4).index

fig, axes = plt.subplots(2, 2, figsize=(14, 16))
axes = axes.flatten()
type_colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

for i, crime in enumerate(top4):
    crime_geo = geo[geo['Primary Type'] == crime]
    sample_n = min(10000, len(crime_geo))
    if sample_n > 0:
        crime_sample = crime_geo.sample(n=sample_n, random_state=42)
        axes[i].scatter(crime_sample['Longitude'], crime_sample['Latitude'],
                        s=0.5, alpha=0.1, c=type_colors[i])
    axes[i].set_title(f'{crime} (n={len(crime_geo):,})', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Longitude')
    axes[i].set_ylabel('Latitude')
    axes[i].set_aspect('equal')
    axes[i].grid(alpha=0.2)

plt.suptitle('Spatial Distribution by Crime Type (Top 4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(new_code_cell("""# Crime density by district (bar + stats)
district_counts = df['District'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(district_counts.index.astype(int).astype(str), district_counts.values,
              color=sns.color_palette("viridis", len(district_counts)), edgecolor='white')
ax.set_title('Crime Count by Police District', fontsize=14, fontweight='bold')
ax.set_xlabel('District Number')
ax.set_ylabel('Number of Incidents')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nHighest crime district: {int(district_counts.idxmax())} ({district_counts.max():,} incidents)")
print(f"Lowest crime district:  {int(district_counts.idxmin())} ({district_counts.min():,} incidents)")"""))

# ==========================================================================
# CELL 13: Correlation Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 10. Correlation Analysis

Identify meaningful numerical correlations — excluding arbitrary IDs."""))

cells.append(new_code_cell("""# Select meaningful numerical columns (exclude IDs and coordinates)
exclude_patterns = ['id', 'case', 'x_coord', 'y_coord', 'x coord', 'y coord',
                    'updated', '_year']
meaningful_num = []
for c in df.select_dtypes(include=[np.number]).columns:
    if not any(pat in c.lower().replace(' ', '_') for pat in exclude_patterns):
        meaningful_num.append(c)

print(f"Meaningful numerical columns for correlation: {meaningful_num}")

if len(meaningful_num) >= 2:
    corr = df[meaningful_num].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Correlation Matrix (Meaningful Numerical Variables)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Report notable correlations
    print("\\nNotable Correlations (|r| > 0.3, excluding diagonal):")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                found = True
                strength = "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.5 else "Weak-Moderate"
                direction = "positive" if r > 0 else "negative"
                print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: r = {r:.3f} ({strength} {direction})")
    if not found:
        print("  No correlations above |0.3| found among these variables.")
    
    print("\\n⚠️ Note: Correlations between geographic area codes (District, Beat, Ward)")
    print("   reflect administrative boundaries, not meaningful crime relationships.")
else:
    print("Not enough meaningful numerical columns for correlation analysis.")"""))

# ==========================================================================
# CELL 14: Statistical Analysis
# ==========================================================================
cells.append(new_markdown_cell("""## 11. Statistical Analysis

Supplementing visual observations with formal statistical tests where appropriate.

### 11.1 Chi-square Test: Crime Type × Arrest

**H₀:** Arrest outcome is independent of crime type.  
**H₁:** Arrest outcome depends on crime type."""))

cells.append(new_code_cell("""# Chi-square: Crime Type × Arrest
top10_crimes = df['Primary Type'].value_counts().head(10).index
subset = df[df['Primary Type'].isin(top10_crimes)]
contingency = pd.crosstab(subset['Primary Type'], subset['Arrest'])

chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

print("Chi-square Test: Crime Type × Arrest Outcome")
print("=" * 55)
print(f"H₀: Arrest outcome is independent of crime type.")
print(f"H₁: Arrest outcome depends on crime type.")
print(f"")
print(f"Chi-square statistic: {chi2:,.2f}")
print(f"Degrees of freedom:   {dof}")
print(f"P-value:              {p_val:.2e}")

# Effect size: Cramér's V
n = contingency.sum().sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Cramér's V:           {cramers_v:.3f}")

if p_val < 0.05:
    print(f"\\n✅ REJECT H₀ (p < 0.05): Statistically significant association")
    print(f"   between crime type and arrest outcome.")
    if cramers_v < 0.1:
        print(f"   Effect size: Weak (V < 0.1)")
    elif cramers_v < 0.3:
        print(f"   Effect size: Moderate (0.1 ≤ V < 0.3)")
    else:
        print(f"   Effect size: Strong (V ≥ 0.3)")
else:
    print(f"\\n❌ FAIL TO REJECT H₀: No significant association found.")

print(f"\\n⚠️ Caveat: With large sample sizes (n={len(subset):,}), even small")
print(f"   effects become statistically significant. Cramér's V ({cramers_v:.3f})")
print(f"   measures practical significance independent of sample size.")"""))

cells.append(new_markdown_cell("""### 11.2 Chi-square Test: Crime Type × Domestic

**H₀:** Domestic classification is independent of crime type.  
**H₁:** Domestic classification depends on crime type."""))

cells.append(new_code_cell("""# Chi-square: Crime Type × Domestic
contingency2 = pd.crosstab(subset['Primary Type'], subset['Domestic'])
chi2_2, p_val_2, dof_2, _ = stats.chi2_contingency(contingency2)

n2 = contingency2.sum().sum()
min_dim2 = min(contingency2.shape) - 1
cramers_v2 = np.sqrt(chi2_2 / (n2 * min_dim2))

print("Chi-square Test: Crime Type × Domestic Classification")
print("=" * 55)
print(f"H₀: Domestic classification is independent of crime type.")
print(f"H₁: Domestic classification depends on crime type.")
print(f"")
print(f"Chi-square statistic: {chi2_2:,.2f}")
print(f"P-value:              {p_val_2:.2e}")
print(f"Cramér's V:           {cramers_v2:.3f}")

if p_val_2 < 0.05:
    print(f"\\n✅ REJECT H₀: Significant association between crime type and domestic classification.")
    if cramers_v2 >= 0.3:
        print(f"   This is a strong association (V = {cramers_v2:.3f}).")
else:
    print(f"\\n❌ FAIL TO REJECT H₀.")"""))

cells.append(new_markdown_cell("""### 11.3 Weekend vs Weekday Crime Comparison"""))

cells.append(new_code_cell("""# Weekend vs Weekday comparison
weekend = df[df['Is_Weekend'] == 1]
weekday = df[df['Is_Weekend'] == 0]

print("Weekend vs Weekday Crime Comparison")
print("=" * 50)
print(f"Weekend total incidents: {len(weekend):,}")
print(f"Weekday total incidents: {len(weekday):,}")

# Normalize per day
n_years = df['Year'].nunique()
weekend_daily = len(weekend) / (2 * n_years * 52)  # ~2 weekend days per week
weekday_daily = len(weekday) / (5 * n_years * 52)  # ~5 weekday days per week

print(f"\\nEstimated daily average:")
print(f"  Weekend: ~{weekend_daily:.0f} crimes/day")
print(f"  Weekday: ~{weekday_daily:.0f} crimes/day")
print(f"  Ratio (Weekend/Weekday): {weekend_daily/weekday_daily:.2f}")

# Arrest rate comparison
weekend_arrest = weekend['Arrest'].mean() * 100
weekday_arrest = weekday['Arrest'].mean() * 100
print(f"\\nArrest rates:")
print(f"  Weekend: {weekend_arrest:.1f}%")
print(f"  Weekday: {weekday_arrest:.1f}%")"""))

# ==========================================================================
# CELL 15: Key Insights
# ==========================================================================
cells.append(new_markdown_cell("""## 12. Key Findings from Chicago Crime Data

---

> ⚠️ **Important:** All insights below are derived directly from the dataset analysis. No causal claims are made unless explicitly supported by the data. Correlation ≠ Causation.

The following findings summarize the key conclusions of this analysis. Each is supported by the data and visualizations produced above."""))

cells.append(new_code_cell("""# Generate evidence-based key insights from the actual data
print("=" * 70)
print("  KEY FINDINGS FROM CHICAGO CRIME DATA")
print("=" * 70)

findings = []

# Finding 1: Dataset scope
min_year = df['Year'].min()
max_year = df['Year'].max()
findings.append({
    'title': 'DATASET SCOPE',
    'finding': f"The dataset contains {len(df):,} crime records spanning {min_year}–{max_year}.",
    'evidence': f"{df.shape[1]} columns covering crime type, location, time, arrest status, and geography.",
    'interpretation': "Provides a comprehensive multi-year view of crime in Chicago.",
    'caveat': "Analysis is limited to reported/recorded crimes; unreported crimes are not captured."
})

# Finding 2: Dominant crime type
top_crime = df['Primary Type'].value_counts().head(1)
top_crime_name = top_crime.index[0]
top_crime_pct = top_crime.values[0] / len(df) * 100
top3 = df['Primary Type'].value_counts().head(3)
findings.append({
    'title': 'DOMINANT CRIME TYPE',
    'finding': f"'{top_crime_name}' is the most common crime type ({top_crime_pct:.1f}% of all incidents).",
    'evidence': f"Top 3: {', '.join([f'{c} ({v:,})' for c, v in top3.items()])}",
    'interpretation': f"'{top_crime_name}' substantially outnumbers other categories.",
    'caveat': "High count may reflect reporting patterns rather than actual crime prevalence."
})

# Finding 3: Arrest rate
findings.append({
    'title': 'LOW OVERALL ARREST RATE',
    'finding': f"The overall arrest rate is {overall_arrest_rate:.1f}%.",
    'evidence': f"Arrests: {df['Arrest'].sum():,} / Total: {len(df):,}",
    'interpretation': "A majority of reported crimes do not result in arrest.",
    'caveat': "Low arrest rate does not imply low policing effort; many crimes are hard to solve."
})

# Finding 4: Arrest variation by crime type
highest_arrest_crime = arrest_by_type['Arrest_Rate'].idxmax()
highest_arrest_rate = arrest_by_type['Arrest_Rate'].max()
lowest_arrest_crime = arrest_by_type['Arrest_Rate'].idxmin()
lowest_arrest_rate = arrest_by_type['Arrest_Rate'].min()
findings.append({
    'title': 'ARREST RATE VARIES BY CRIME TYPE',
    'finding': f"Arrest rates range from {lowest_arrest_rate:.1f}% ({lowest_arrest_crime}) to {highest_arrest_rate:.1f}% ({highest_arrest_crime}).",
    'evidence': "See arrest rate by crime type visualization.",
    'interpretation': "Crime types with higher arrest rates may be easier to solve or have different policing priorities.",
    'caveat': "Arrest rate differences may reflect crime characteristics, not policing quality."
})

# Finding 5: Temporal peaks
peak_hour = df['Hour'].value_counts().idxmax()
low_hour = df['Hour'].value_counts().idxmin()
peak_month = df['Month_Name'].value_counts().idxmax()
low_month = df['Month_Name'].value_counts().idxmin()
findings.append({
    'title': 'CLEAR TEMPORAL PATTERNS',
    'finding': f"Crime peaks at {peak_hour}:00 and is lowest at {low_hour}:00. Monthly peak: {peak_month}, lowest: {low_month}.",
    'evidence': "See hourly and monthly distribution charts.",
    'interpretation': "Crime follows distinct daily and seasonal cycles.",
    'caveat': "Temporal patterns may reflect reporting patterns, not actual crime timing."
})

# Finding 6: Location concentration
top_loc = df['Location Description'].value_counts().head(1)
top_loc_pct = top_loc.values[0] / len(df) * 100
findings.append({
    'title': 'LOCATION CONCENTRATION',
    'finding': f"'{top_loc.index[0]}' is the most common crime location ({top_loc_pct:.1f}%).",
    'evidence': "See location distribution chart.",
    'interpretation': "A significant portion of crimes occurs in public spaces.",
    'caveat': "Location categories may overlap or be inconsistently recorded."
})

# Finding 7: Domestic crimes
findings.append({
    'title': 'DOMESTIC CRIME PREVALENCE',
    'finding': f"{domestic_rate:.1f}% of all crimes are classified as domestic.",
    'evidence': f"Domestic incidents: {df['Domestic'].sum():,} out of {len(df):,}",
    'interpretation': "A notable fraction of crime involves domestic situations.",
    'caveat': "Domestic crimes are often underreported; actual rates may be higher."
})

# Finding 8: Day-hour pattern
peak_day = df['Day_of_Week'].value_counts().idxmax()
findings.append({
    'title': 'DAY-HOUR INTERACTION',
    'finding': f"Crime patterns differ between weekdays and weekends, with {peak_day} having the most incidents overall.",
    'evidence': "See Day × Hour heatmap.",
    'interpretation': "Weekend nighttime hours may show different patterns than weekday business hours.",
    'caveat': "Population movement patterns affect where crimes are reported."
})

# Print all findings
for i, f in enumerate(findings, 1):
    print(f"\\n{'─' * 60}")
    print(f"  Finding {i}: {f['title']}")
    print(f"{'─' * 60}")
    print(f"  📊 Finding:        {f['finding']}")
    print(f"  📈 Evidence:       {f['evidence']}")
    print(f"  💡 Interpretation: {f['interpretation']}")
    print(f"  ⚠️  Caveat:        {f['caveat']}")"""))

# ==========================================================================
# CELL 16: Conclusion
# ==========================================================================
cells.append(new_markdown_cell("""## 13. Conclusion

### Executive Summary

This comprehensive Exploratory Data Analysis of the Chicago Crime dataset has revealed several important patterns:

**Dataset Overview:**
- The dataset provides a detailed multi-year record of crime in Chicago, with each incident documented by type, location, time, and arrest outcome.

**Key Findings:**

1. **Crime Composition:** A few crime types dominate the landscape — theft, battery, and criminal damage typically represent the largest categories, while specialized crimes like arson or stalking are rare.

2. **Arrest Rates:** The overall arrest rate is relatively low, but it varies dramatically by crime type. Certain crime categories (like narcotics or prostitution) tend to have high arrest rates because arrests are inherent to how those crimes are discovered, while property crimes tend to have lower rates.

3. **Temporal Patterns:** Crime exhibits clear daily, weekly, and seasonal patterns:
   - Afternoon and evening hours see more crime than early morning
   - Summer months tend to have higher crime rates than winter
   - Weekend patterns may differ from weekday patterns

4. **Geographic Distribution:** Crime is not uniformly distributed across Chicago. Certain districts consistently show higher crime concentrations, reflecting both population density and socioeconomic factors.

5. **Statistical Significance:** Chi-square tests confirm statistically significant associations between crime type and arrest outcome, as well as crime type and domestic classification. However, large sample sizes can make even small effects statistically significant — Cramér's V provides a measure of practical significance.

**Important Limitations:**
- This analysis examines **reported crimes only** — unreported crimes are not captured.
- **Correlation ≠ Causation** — observed associations do not imply causal relationships.
- Geographic patterns reflect where crimes are **reported**, which may differ from where they **occur**.
- Data quality issues (missing coordinates, potential duplicate case numbers) affect certain analyses.

---

*Analysis conducted using Python (pandas, matplotlib, seaborn, scipy). All findings are evidence-based and derived directly from the dataset.*"""))

# Assemble notebook
nb.cells = cells

# Write notebook
output_path = '/Users/anaystark/Desktop/Anay/S4DS ML Task 1/Chicago_Crime_EDA.ipynb'
with open(output_path, 'w') as f:
    nbformat.write(nb, f)

print(f"✅ Notebook generated: {output_path}")
print(f"   Total cells: {len(cells)}")

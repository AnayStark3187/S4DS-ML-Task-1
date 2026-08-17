# 🏙️ Chicago Crime Dataset — Exploratory Data Analysis

A comprehensive **Exploratory Data Analysis (EDA)** of reported crime incidents in Chicago from **2021–2025**, focusing on crime composition, temporal patterns, geographic distribution, arrest outcomes, domestic incidents, and statistical relationships.

---

## 📌 Overview

This project analyzes the Chicago Crime dataset to identify meaningful patterns in:

* Crime types and descriptions
* Crime locations and districts
* Arrest outcomes
* Domestic crime incidents
* Daily, monthly, and yearly crime patterns
* Geographic distribution across Chicago
* Relationships between crime characteristics
* Statistical associations between crime type, arrest, and domestic classification

The analysis is entirely data-driven, with interpretations based on the observations and statistical tests performed in the notebook.

---

## 🎯 Objectives

The primary objectives of this project are to:

1. Understand the composition and distribution of reported crimes in Chicago.
2. Identify the most common crime types and descriptions.
3. Analyze arrest and domestic crime rates.
4. Discover temporal patterns across years, months, days, and hours.
5. Examine geographic crime concentration across districts and community areas.
6. Investigate relationships between crime type, location, time, and arrest outcomes.
7. Use statistical tests to determine whether observed relationships are statistically significant.
8. Assess the quality and completeness of the dataset.

---

## 📊 Dataset

The dataset contains individual reported crime incidents with information about:

| Category       | Variables                                               |
| -------------- | ------------------------------------------------------- |
| Identification | `ID`, `Case Number`                                     |
| Crime          | `Primary Type`, `Description`, `IUCR`, `FBI Code`       |
| Location       | `Block`, `Location Description`, `Location`             |
| Outcome        | `Arrest`, `Domestic`                                    |
| Administrative | `Beat`, `District`, `Ward`, `Community Area`            |
| Geography      | `Latitude`, `Longitude`, `X Coordinate`, `Y Coordinate` |
| Time           | `Date`, `Year`, `Updated On`                            |

### Dataset Size

After data cleaning, the analysis contains:

* **116,687 crime records**
* **30 columns**
* **Time period:** 2021–2025
* **31 unique crime types**
* **302 unique crime descriptions**
* **129 location categories**

> The original dataset contained 120,759 records and 22 columns before date-based cleaning and feature engineering.

---

## 🛠️ Technologies & Libraries

The project is implemented in Python using:

* **Python**
* **Pandas** — data manipulation and analysis
* **NumPy** — numerical operations
* **Matplotlib** — data visualization
* **Seaborn** — statistical visualization
* **SciPy** — statistical testing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

---

## 🔄 Analysis Workflow

The notebook follows a structured EDA pipeline:

```text
Load Dataset
     ↓
Data Overview
     ↓
Data Quality Assessment
     ↓
Data Cleaning & Feature Engineering
     ↓
Univariate Analysis
     ↓
Bivariate Analysis
     ↓
Multivariate Analysis
     ↓
Geographic Analysis
     ↓
Correlation Analysis
     ↓
Statistical Analysis
     ↓
Key Findings & Conclusion
```

---

## 🧹 Data Cleaning & Feature Engineering

The `Date` column is converted into a datetime object and used to generate additional temporal features:

* `Month`
* `Month_Name`
* `Day`
* `Day_of_Week`
* `Day_of_Week_Num`
* `Hour`
* `Quarter`
* `Is_Weekend`

Records with midnight timestamps (`00:00:00`) were removed as part of the preprocessing performed in the notebook.

### Data Quality

The analysis identified:

* No completely duplicated rows
* Unique IDs for all cleaned records
* A small number of duplicate case numbers
* Missing geographic information for a small portion of records
* No invalid geographic coordinates outside the defined Chicago bounds

The primary missing values were associated with:

* `X Coordinate`
* `Y Coordinate`
* `Latitude`
* `Longitude`
* `Location`
* `Location Description`
* `Community Area`

Missing geographic values were handled contextually rather than simply dropping all affected records.

---

# 📈 Exploratory Analysis

## 1. Crime Type Distribution

The dataset contains **31 crime types**.

The most common categories are:

| Rank | Crime Type          | Incidents |
| ---: | ------------------- | --------: |
|    1 | THEFT               |    22,867 |
|    2 | BATTERY             |    21,939 |
|    3 | CRIMINAL DAMAGE     |    10,914 |
|    4 | ASSAULT             |    10,164 |
|    5 | OTHER OFFENSE       |     7,870 |
|    6 | MOTOR VEHICLE THEFT |     7,788 |
|    7 | WEAPONS VIOLATION   |     7,325 |
|    8 | NARCOTICS           |     6,828 |
|    9 | DECEPTIVE PRACTICE  |     6,120 |
|   10 | ROBBERY             |     3,834 |

**THEFT** is the most frequently recorded crime, accounting for approximately **19.6%** of incidents.

---

## 2. Crime Descriptions

The dataset contains **302 unique crime descriptions**.

The most frequent descriptions include:

* `SIMPLE`
* `DOMESTIC BATTERY SIMPLE`
* `$500 AND UNDER`
* `OVER $500`
* `RETAIL THEFT`
* `AUTOMOBILE`
* `UNLAWFUL POSSESSION - HANDGUN`
* `TO VEHICLE`
* `TO PROPERTY`

---

## 3. Arrest Analysis

The overall arrest rate is approximately:

### **33.5%**

| Arrest Status | Incidents | Percentage |
| ------------- | --------: | ---------: |
| No Arrest     |    77,610 |     66.51% |
| Arrest        |    39,077 |     33.49% |

The analysis also examines how arrest rates differ between crime types.

Arrest rates vary substantially, ranging from approximately:

* **9.0%** for `DECEPTIVE PRACTICE`
* **98.9%** for `NARCOTICS`

These differences should not automatically be interpreted as differences in policing effectiveness, since arrest rates can depend strongly on the characteristics and detection mechanisms of different crimes.

---

## 4. Domestic Crime

Approximately **19.4%** of recorded incidents are classified as domestic.

| Classification | Incidents | Percentage |
| -------------- | --------: | ---------: |
| Non-Domestic   |    94,080 |     80.63% |
| Domestic       |    22,607 |     19.37% |

---

## 5. Crime Locations

There are **129 unique location categories**.

The most common locations include:

| Location           | Incidents | Percentage |
| ------------------ | --------: | ---------: |
| STREET             |    32,640 |      28.0% |
| APARTMENT          |    20,901 |      17.9% |
| RESIDENCE          |    13,072 |      11.2% |
| SIDEWALK           |     7,701 |       6.6% |
| SMALL RETAIL STORE |     3,993 |       3.4% |

`STREET` is the most frequently recorded crime location.

---

# 🕒 Temporal Analysis

Crime patterns were analyzed across:

* Year
* Month
* Day of week
* Hour
* Quarter
* Weekend vs weekday

### Incidents by Year

| Year | Incidents |
| ---: | --------: |
| 2021 |    20,127 |
| 2022 |    22,194 |
| 2023 |    24,803 |
| 2024 |    25,472 |
| 2025 |    24,091 |

### Hourly Pattern

* **Highest crime hour:** 12:00 — 6,885 incidents
* **Lowest crime hour:** 05:00 — 1,911 incidents

The analysis also examines seasonal patterns and interactions between crime type and time.

---

# 🗺️ Geographic Analysis

The dataset contains latitude and longitude information that allows spatial crime analysis.

After cleaning:

* **115,061 records** had valid coordinates
* Approximately **98.6%** of records available for geographic analysis
* **1,626 records** lacked usable coordinates

Geographic analysis includes:

* Crime distribution by district
* Spatial crime distribution
* District-level crime concentration
* Crime type distribution across districts

The district with the highest number of recorded incidents was **District 11**, with **8,014 incidents**.

---

# 🔬 Multivariate Analysis

The project goes beyond single-variable analysis by examining combinations of variables.

### Day × Hour Heatmap

A crime heatmap was created to investigate how crime frequency varies simultaneously across:

* Day of week
* Hour of day

Some of the highest day-hour combinations include:

* Wednesday at 12:00 — 1,042 incidents
* Thursday at 12:00 — 1,041 incidents
* Monday at 12:00 — 1,031 incidents
* Friday at 15:00 — 1,028 incidents
* Tuesday at 15:00 — 1,018 incidents

Other multivariate analyses include:

* Crime Type × Hour × Arrest Rate
* Crime Type × Month
* District × Crime Type
* Crime Type × Day of Week

---

# 📐 Correlation Analysis

Correlation analysis was performed on meaningful numerical and engineered variables while excluding arbitrary identifiers.

Notable relationships include:

| Variables                 | Correlation |
| ------------------------- | ----------: |
| Beat ↔ District           |       1.000 |
| Ward ↔ Latitude           |       0.712 |
| District ↔ Ward           |       0.651 |
| Beat ↔ Ward               |       0.650 |
| Community Area ↔ Latitude |      -0.753 |
| Beat ↔ Latitude           |       0.629 |
| District ↔ Latitude       |       0.630 |

These correlations primarily reflect the geographic and administrative structure of Chicago rather than causal relationships.

---

# 🧪 Statistical Analysis

## Crime Type × Arrest

A chi-square test was performed to determine whether arrest outcome is independent of crime type.

* **Chi-square statistic:** 32,075.63
* **Degrees of freedom:** 9
* **p-value:** < 0.001
* **Cramér's V:** 0.551

### Result

The null hypothesis is rejected.

There is a statistically significant and strong association between **crime type and arrest outcome**.

---

## Crime Type × Domestic Classification

A second chi-square test examined the relationship between crime type and domestic classification.

* **Chi-square statistic:** 29,179.38
* **p-value:** < 0.001
* **Cramér's V:** 0.526

### Result

The null hypothesis is rejected.

There is a statistically significant and strong association between **crime type and domestic classification**.

---

## Weekend vs Weekday

The analysis compares crime frequency and arrest rates between weekends and weekdays.

| Metric                | Weekend | Weekday |
| --------------------- | ------: | ------: |
| Total incidents       |  33,679 |  83,008 |
| Approx. daily average |      65 |      64 |
| Arrest rate           |   33.6% |   33.4% |

The weekend-to-weekday daily incident ratio is approximately **1.01**, indicating very similar average daily crime counts.

---

# 🔎 Key Findings

### 1. Theft is the dominant crime type

`THEFT` accounts for approximately **19.6%** of recorded incidents, followed by `BATTERY` and `CRIMINAL DAMAGE`.

### 2. Most reported crimes do not result in an arrest

The overall arrest rate is approximately **33.5%**, meaning roughly two-thirds of recorded incidents have no associated arrest.

### 3. Arrest rates vary significantly by crime type

Arrest rates range from approximately **9.0% to 98.9%**, demonstrating substantial differences between crime categories.

### 4. Crime exhibits temporal patterns

Crime frequency varies considerably by hour, day, month, and year, with **12:00** being the highest recorded crime hour in this dataset.

### 5. Street incidents dominate location categories

`STREET` represents approximately **28%** of recorded crime locations.

### 6. Domestic incidents form a significant portion of reported crime

Approximately **19.4%** of incidents are classified as domestic.

### 7. Crime is geographically concentrated

Crime is not uniformly distributed across Chicago's districts, with some districts recording substantially more incidents than others.

### 8. Crime type is strongly associated with arrest and domestic classification

The chi-square tests and Cramér's V values indicate strong statistical associations between crime type and both arrest outcome and domestic classification.

---

# 📁 Project Structure

```text
Chicago-Crime-EDA/
│
├── Chicago_Crime_EDA.ipynb
├── chicago_crime_data.csv
└── README.md
```

> The dataset may be omitted from the repository if its size or redistribution restrictions make that preferable. The notebook can instead load the dataset from its configured source.

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd Chicago-Crime-EDA
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Chicago_Crime_EDA.ipynb
```

### 4. Run the notebook

Execute the cells sequentially to reproduce the data cleaning, analysis, visualizations, statistical tests, and findings.

---

# ⚠️ Limitations

This analysis has several important limitations:

* The dataset represents **reported/recorded crimes**, not all crimes that actually occurred.
* Missing geographic information prevents some incidents from being included in spatial analysis.
* Correlation and statistical association **do not imply causation**.
* Crime counts may be affected by reporting behavior and data collection practices.
* Differences in arrest rates between crime types do not necessarily indicate differences in police effectiveness.
* Geographic patterns represent where incidents are recorded and may not perfectly represent where crimes actually occurred.
* Potential duplicate case numbers exist even though completely duplicated rows were not found.
* Large datasets can produce statistically significant relationships even when practical effects may be more limited, which is why **Cramér's V** was also considered.

---

# 💡 Conclusion

This project demonstrates a complete EDA workflow for a large real-world crime dataset. By combining descriptive statistics, data visualization, temporal analysis, geographic analysis, correlation analysis, and hypothesis testing, the analysis provides a structured view of how crime varies across **type, location, time, district, arrest outcome, and domestic classification**.

The results demonstrate that Chicago's recorded crime patterns are highly heterogeneous: a relatively small number of crime categories account for a large proportion of incidents, crime frequency varies substantially across time and geography, and crime type is strongly associated with both arrest and domestic classification.

Most importantly, the analysis distinguishes **statistical association from causation** and acknowledges the limitations inherent in using reported crime data.

---

## 👨‍💻 Tools Used

**Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Jupyter Notebook**

---

*All findings presented in this README are derived from the analysis performed in the accompanying notebook.*

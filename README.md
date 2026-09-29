# 🏅 OLYMPIA: Olympic Performance Analytics & Intelligence System

A comprehensive, offline-first Data Warehousing and Data Mining analytics platform built on the complete historical Olympic Games dataset (1896 – 2024).

---

## 1. Project Title
**OLYMPIA: Olympic Performance Analytics & Intelligence System**

---

## 2. Problem Statement
For over a century, the Olympic Games have produced vast quantities of multidimensional performance data covering thousands of athletes, sports, events, and participating nations. However, raw Olympic datasets typically present challenges:
- Unstructured rosters, non-awarded/voided event records, and encoding irregularities.
- Lack of centralized dimensional modeling (Star Schema) for fast analytical processing.
- Disconnected analytical scripts that fail to provide an integrated decision-support environment for OLAP, regression forecasting, classification, clustering, and association rule discovery.

OLYMPIA solves this problem by integrating a Star Schema Data Warehouse, an automated ETL pipeline, and interactive Data Mining algorithms into a unified dark-themed analytical dashboard.

---

## 3. Objective
To build an end-to-end, reproducible Data Warehousing and Data Mining system that demonstrates:
1. **Data Warehousing & Star Schema Modeling** (Centralized Fact table with 5 Dimension tables).
2. **Automated ETL Pipeline** (Extract, Clean, Normalize, and Load into SQLite).
3. **Multidimensional OLAP Operations** (Roll-Up, Drill-Down, Slice, Dice, and Pivot).
4. **Data Visualization Laboratory** (Bar, Line, Scatter, Histogram, Box Plot).
5. **Data Preprocessing & Quality Analysis** (Encoding, Scaling, PCA Dimensionality Reduction).
6. **Supervised Regression Modeling** (Predicting national medal counts with zero future data leakage).
7. **Supervised Classification Modeling** (Decision Tree vs. Naive Bayes for podium outcome prediction).
8. **Unsupervised Clustering Modeling** (K-Means with Elbow Analysis & Agglomerative Hierarchical Clustering with Dendrograms).
9. **Association Rule Mining** (Apriori market-basket analysis on multi-sport winning affinities).
10. **Interactive Decision-Support Interface** (Streamlit dashboard with dynamic filtering and viva explanations).

---

## 4. Dataset Description
- **Source File:** `all_olympic_medalists.csv`
- **Total Raw Records:** 20,247 rows × 10 columns
- **Temporal Span:** 1896 (Athens) to 2024 (Paris) across 53 Olympic editions.
- **Attributes:**
  - `season`: Summer (16,747) or Winter (3,500).
  - `year`: Competition year (1896 – 2024).
  - `medal`: Gold, Silver, Bronze.
  - `country_code`: 3-letter National Olympic Committee (NOC) code (158 unique codes).
  - `country`: Full country name (168 historical names).
  - `athletes`: Competing athlete names or team representation.
  - `games`: Host city edition (e.g., `1896 Athens`, `2024 Paris`).
  - `sport`: Sporting discipline (77 unique disciplines).
  - `event_gender`: Men's (13,105), Women's (5,934), Mixed (1,208).
  - `event_name`: Specific athletic event (e.g., `100m`, `marathon`, `team`).

---

## 5. Technologies Used
- **Programming Language:** Python 3.10+ (tested on Python 3.14)
- **Data Engineering:** Pandas, NumPy
- **Relational Database:** SQLite3 (`database/olympics.db`)
- **Machine Learning & Mining:** Scikit-learn, MLxtend, Scipy
- **Data Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib
- **Web Dashboard:** Streamlit
- **Quality & Testing:** Python `unittest` framework

---

## 6. System Architecture

```
+--------------------------------------------------------------------------+
|                            SOURCE LAYER                                  |
|               Raw CSV: data/raw/all_olympic_medalists.csv               |
+--------------------------------------------------------------------------+
                                    |
                                    v  [etl/extract.py]
+--------------------------------------------------------------------------+
|                         TRANSFORMATION LAYER                             |
|  - Deduplication (dropped 3 duplicates)                                  |
|  - Voided event filtering (dropped 4 non-awarded NaN medals)             |
|  - Missing athlete imputation ('Team / Not Listed')                      |
|  - Medal point weighting (Gold=3, Silver=2, Bronze=1)                   |
|  Output: data/processed/cleaned_olympic_medalists.csv (20,240 records)   |
+--------------------------------------------------------------------------+
                                    |
                                    v  [etl/load.py]
+--------------------------------------------------------------------------+
|                     DATA WAREHOUSE (STAR SCHEMA)                         |
|                     database/olympics.db (SQLite3)                       |
|   Dimensions: dim_game, dim_country, dim_sport, dim_event, dim_medal     |
|   Fact Table: fact_medal (20,240 facts with indexed foreign keys)        |
+--------------------------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|    OLAP & REPORTING   |                       |    DATA MINING / ML   |
| - SLICE & DICE        |                       | - Linear Regression   |
| - ROLL-UP & DRILL-DOWN|                       | - Decision Tree / NB  |
| - PIVOT MATRICES      |                       | - K-Means & Dendrogram|
| - Interactive Plotly  |                       | - Apriori Rule Mining |
+-----------------------+                       +-----------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                      STREAMLIT ANALYTICS DASHBOARD                       |
|               11 Dedicated Modules • Dark Analytics Theme               |
+--------------------------------------------------------------------------+
```

---

## 7. Database Schema (Star Schema)

The dimensional data warehouse adheres to a strict Star Schema stored in `database/olympics.db`:

```
                         +-----------------------+
                         |       dim_game        |
                         +-----------------------+
                         | PK: game_id           |
                         |     year              |
                         |     season            |
                         |     games (UNIQUE)    |
                         +-----------+-----------+
                                     |
                                     | 1:N
+-----------------------+            |            +-----------------------+
|      dim_country      |            |            |       dim_sport       |
+-----------------------+            |            +-----------------------+
| PK: country_id        |            |            | PK: sport_id          |
|     country_code      |            v            |     sport_name (UQ)   |
|     country_name (UQ) |---+ +-------------+ +---|                       |
+-----------------------+   | |             | |   +-----------+-----------+
                            | |             | |               |
                            v v             v v               | 1:N
                         +-----------------------+            v
                         |      fact_medal       |   +-----------------------+
                         +-----------------------+   |       dim_event       |
                         | PK: medal_fact_id     |   +-----------------------+
                         | FK: game_id           |   | PK: event_id          |
                         | FK: country_id        |-->| FK: sport_id          |
                         | FK: sport_id          |   |     event_name        |
                         | FK: event_id          |   |     event_gender      |
                         | FK: medal_id          |<--+-----------------------+
                         |     athletes          |
                         +-----------+-----------+
                                     ^
                                     | 1:N
                         +-----------+-----------+
                         |       dim_medal       |
                         +-----------------------+
                         | PK: medal_id          |
                         |     medal_name (UQ)   |
                         |     medal_points      |
                         +-----------------------+
```

### Table Specifications:
| Table | Type | Rows | Primary Key | Foreign Keys |
|---|---|---|---|---|
| `dim_game` | Dimension | 53 | `game_id` | None |
| `dim_country` | Dimension | 168 | `country_id` | None |
| `dim_sport` | Dimension | 77 | `sport_id` | None |
| `dim_event` | Dimension | 1,006 | `event_id` | `sport_id` → `dim_sport` |
| `dim_medal` | Dimension | 3 | `medal_id` | None |
| `fact_medal` | Fact | 20,240 | `medal_fact_id` | `game_id`, `country_id`, `sport_id`, `event_id`, `medal_id` |

---

## 8. ETL Process
The ETL pipeline (`etl/`) guarantees repeatability and idempotency:
1. **Extract (`etl/extract.py`):** Loads raw records from `data/raw/all_olympic_medalists.csv`.
2. **Transform (`etl/transform.py`):**
   - Strips leading and trailing whitespace from string attributes.
   - Drops 3 exact duplicate records (1900 Paris sailing).
   - Drops 4 non-awarded voided events where medals were NaN.
   - Imputes 423 omitted team rosters with `'Team / Not Listed'`.
   - Computes internal analytical scoring: `Gold = 3`, `Silver = 2`, `Bronze = 1`.
   - Generates verified clean CSV in `data/processed/cleaned_olympic_medalists.csv`.
3. **Load (`etl/load.py`):**
   - Executes DDL schema (`database/schema.sql`).
   - Inserts dimensions and generates foreign-key mappings.
   - Populates `fact_medal` with 20,240 records.
   - Verifies integrity via `PRAGMA foreign_key_check`.

---

## 9. OLAP Operations
Implemented in `analytics/olap.py` and visualized in the dashboard:
- **SLICE:** Isolates one dimension value (e.g. `Year = 2024` or `Country = 'India'`).
- **DICE:** Filters multiple dimensions simultaneously (e.g. `Years: 2016-2024` AND `Season: Summer` AND `Selected Nations`).
- **ROLL-UP:** Summarizes data up conceptual hierarchies (e.g., `Event` → `Sport` → `Country`).
- **DRILL-DOWN:** Navigates from aggregate performance down to specific sport and event breakdowns.
- **PIVOT:** Cross-tabulates country performance across medal classes (Rows = Country, Columns = Gold/Silver/Bronze/Total).

---

## 10. Visualization Experiment
Adheres to the Dark Analytics theme with responsive Plotly figures:
1. **Bar Chart:** Categorical comparisons of top nations and sports.
2. **Line Chart:** Temporal trajectories of medal counts across Olympic years.
3. **Scatter Plot:** Bivariate correlation between Gold medals and total medal points.
4. **Histogram:** Statistical frequency distribution of national medal totals.
5. **Box Plot:** Outlier detection and quartile spread of medals awarded per edition across disciplines.

---

## 11. Machine Learning Algorithms

### A. Linear Regression (`analytics/regression.py`)
- **Objective:** Predict total medals won by a country at Olympic edition $t$.
- **Features:** Lagged historical performance from edition $t-1$: `prev_total_medals`, `prev_gold_medals`, `prev_silver_medals`, `prev_bronze_medals`, `prev_medal_points`.
- **Chronological Split:** Train on past games ($\le 2012$), test on unseen future games ($> 2012$).
- **Results:** $R^2 \approx 0.881$, $\text{MAE} \approx 3.80$, $\text{RMSE} \approx 5.79$.

### B. Classification: Decision Tree vs. Naive Bayes (`analytics/decision_tree.py`, `analytics/naive_bayes.py`)
- **Objective:** Predict podium medal type (Gold, Silver, Bronze) from event contexts (`season`, `year`, `country`, `sport`, `event_gender`).
- **Data Leakage Prevention:** `medal_points` is strictly excluded from feature sets.
- **Decision Tree:** Recursive Gini/Entropy splitting with pruning depth controls.
- **Naive Bayes:** Probabilistic classification using Categorical Naive Bayes.
- **Observed Accuracy:** $\approx 36.6\% - 36.9\%$ (honest baseline for unpredictable athletic outcomes).

---

## 12. Clustering

### A. K-Means Clustering (`analytics/kmeans.py`)
- Standardizes country performance vectors (`gold`, `silver`, `bronze`, `total_medals`, `medal_points`).
- Dynamically assigns nations into $K$ algorithmic performance clusters.
- Includes Elbow Curve analysis (Inertia vs. $K$) to identify optimal grouping.

### B. Agglomerative Hierarchical Clustering (`analytics/hierarchical.py`)
- Builds bottom-up agglomerative cluster hierarchies using Ward, Complete, or Average linkage.
- Generates dendrogram tree visualizations depicting distances between national medal profiles.

---

## 13. Association Rule Mining (Apriori)
- **Module:** `analytics/apriori.py`
- **Transaction Definition:** Country + Olympic Games Edition (e.g. `USA (2024 Paris)`).
- **Basket Items:** Distinct sports in which medals were won.
- **Mined Metrics:** Support $P(A \cap B)$, Confidence $P(B|A)$, and Lift $\frac{P(B|A)}{P(B)}$.
- **Sample Rule:** $\text{Cycling Track} \implies \text{Rowing}$ ($\text{Support}=8.0\%$, $\text{Confidence}=64.7\%$, $\text{Lift}=3.35$).

---

## 14. How to Install

1. Clone or navigate to the project directory:
```bash
cd d:\coding\Olympia
```

2. Create and activate a Python virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

---

## 15. How to Run

### Method 1: Using Windows Launcher Script
Double click or execute from terminal:
```cmd
run.bat
```

### Method 2: Manual Step-by-Step
1. Run ETL to generate and populate the Star Schema:
```bash
python -m etl.load
```

2. Run automated tests to verify system integrity:
```bash
python -m unittest discover -s tests
```

3. Launch the Streamlit application:
```bash
streamlit run dashboard/app.py
```

---

## 16. Project Folder Structure

```
olympia/
│
├── data/
│   ├── raw/
│   │   └── all_olympic_medalists.csv       # Original dataset (source of truth)
│   └── processed/
│       └── cleaned_olympic_medalists.csv   # Cleaned transformed dataset
│
├── database/
│   ├── olympics.db                         # SQLite Star Schema database
│   └── schema.sql                          # DDL Star Schema definition
│
├── etl/
│   ├── __init__.py
│   ├── extract.py                          # Extraction module
│   ├── transform.py                        # Cleaning and normalization
│   └── load.py                             # Idempotent warehouse loader
│
├── warehouse/
│   ├── __init__.py
│   └── warehouse.py                        # Star join query engine & schema access
│
├── analytics/
│   ├── __init__.py
│   ├── olap.py                             # Roll-up, Drill-down, Slice, Dice, Pivot
│   ├── preprocessing.py                    # Quality audits, encoding, scaling, PCA
│   ├── visualization.py                    # Plotly chart generators
│   ├── regression.py                       # Chronological linear regression
│   ├── decision_tree.py                    # Decision tree classifier
│   ├── naive_bayes.py                      # Categorical Naive Bayes
│   ├── kmeans.py                           # K-Means & Elbow method
│   ├── hierarchical.py                     # Agglomerative clustering & Dendrogram
│   └── apriori.py                          # Market basket association rule mining
│
├── dashboard/
│   ├── app.py                              # Streamlit entry point
│   ├── pages/
│   │   ├── dashboard.py                    # Main dashboard & India spotlight
│   │   ├── medal_analysis.py               # Medal analytics & 5 chart types
│   │   ├── country_analysis.py             # Nation deep dive & comparison
│   │   ├── sport_analysis.py               # Sport discipline intelligence
│   │   ├── olap_analysis.py                # Interactive OLAP laboratory
│   │   ├── preprocessing.py                # Preprocessing & quality lab
│   │   ├── regression.py                   # Linear regression forecasting
│   │   ├── classification.py               # Decision tree vs Naive Bayes
│   │   ├── clustering.py                   # K-Means & Hierarchical clustering
│   │   ├── association_rules.py            # Apriori sport affinity rules
│   │   └── warehouse.py                    # Star schema visualizer & inspector
│   └── components/
│       ├── charts.py                       # Chart display wrapper
│       ├── filters.py                      # Global sidebar filter controls
│       └── metrics.py                      # KPI card renderers & viva notes
│
├── tests/
│   ├── test_etl.py                         # ETL pipeline unit tests
│   ├── test_database.py                    # Database schema & FK tests
│   └── test_analytics.py                   # OLAP and ML algorithms tests
│
├── requirements.txt                        # Clean dependency specifications
├── README.md                               # Comprehensive documentation
├── run.bat                                 # Windows execution script
└── .gitignore                              # Git exclusion rules
```

---

## 17. Expected Output
- **Console / Terminal:** Clean extraction, transform, and load logs showing 20,240 fact records loaded with zero FK violations.
- **Unit Tests:** `Ran 18 tests in ~2.5s ... OK`.
- **Browser:** Dark-themed analytics dashboard accessible at `http://localhost:8501`, featuring 11 interactive pages, responsive cards, Plotly charts, and downloadable tables.

---

## 18. Limitations
1. **Historical Team Rosters:** In team competitions (e.g. 1996 Basketball, historical Lacrosse), individual athlete rosters were omitted in the source CSV and are standardized as `'Team / Not Listed'`.
2. **Podium Unpredictability:** Predicting the exact medal color (Gold vs. Silver vs. Bronze) from pre-game context alone has an inherent ceiling ($\approx 37\%$), as athletic podium finishes are determined by fractions of a second or point.
3. **Historical Country Codes:** Some historical entities share modern NOC codes (e.g. `OAR` vs. `ROC`, `CIV` for Ivory Coast vs. Côte d'Ivoire); the warehouse preserves historical naming fidelity through `(country_code, country_name)` pairs.

---

## 19. Future Scope
1. **Athlete Biometrics Integration:** Incorporating athlete physical measurements (height, weight, age) for deeper athlete-level machine learning models.
2. **Economic & Demographic Features:** Enriching country dimensions with GDP, population, and sports expenditure to enhance regression accuracy.
3. **Time-Series Neural Forecasting:** Integrating LSTM / Prophet models for multi-year nation performance forecasting.

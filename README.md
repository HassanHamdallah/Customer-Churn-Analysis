## Overview

This project performs end-to-end data preprocessing and exploratory data analysis (EDA) on a simulated customer dataset to identify early indicators of customer churn in a telecommunications context.

Customer churn — when customers abandon a service — is a critical business problem. Early identification of at-risk customers enables targeted retention strategies, reducing revenue loss and improving loyalty.

---

## Dataset

**File:** `customer_data.csv`  
1,000 customer records across 8 features:

| Feature | Type | Description | Range / Values |
|---|---|---|---|
| CustomerID | Categorical | Unique customer identifier | CUST0000–CUST0999 |
| Age | Numerical | Customer age in years | 18–65 (mean ~41) |
| Gender | Categorical | Gender (encoded) | 0 = Male, 1 = Female |
| Income | Numerical | Annual income | $20,000–$180,000 (mean ~$80k) |
| Tenure | Numerical | Years with the company | 0–20 (mean ~9.8) |
| ProductType | Categorical | Product tier | 0 = Type A (70%), 1 = Type B (30%) |
| SupportCalls | Numerical | Number of support calls made | Varies |
| ChurnStatus | **Target** | Whether customer churned | 0 = No Churn, 1 = Churned |

> Target distribution: **70% not churned**, 30% churned.

---

## Preprocessing Pipeline

### 1. Handling Missing Values
Missing values were detected with `.isnull().sum()`:

| Column | Missing Count |
|---|---|
| Age | 175 |
| Income | 172 |
| Tenure | 175 |
| SupportCalls | 171 |

**Strategy:** Imputed all missing numeric values with the column **median**, preserving all 1,000 records.

### 2. Outlier Detection & Removal
Outliers detected using the **Z-score method** (|Z| > 3):
- Income: 50 outliers
- SupportCalls: 70 outliers
- Age, Tenure: 0 outliers

**Strategy:** **Winsorization (Capping)** at the 1st and 98th percentiles. A follow-up Z-score check confirmed 0 outliers remained.

### 3. Feature Scaling
**Min-Max Normalization** applied to all numerical features, scaling to [0, 1]:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

---

## Exploratory Data Analysis

### Summary Statistics (post-normalization)

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| Age | 0.511 | 0.290 | 0.00 | 0.500 | 1.00 |
| Income | 0.442 | 0.259 | 0.00 | 0.444 | 1.00 |
| Tenure | 0.505 | 0.313 | 0.00 | 0.500 | 1.00 |
| SupportCalls | 0.424 | 0.287 | 0.00 | 0.408 | 1.00 |

### Categorical Feature Summary

| Feature | Category | Count | % |
|---|---|---|---|
| Gender | 0 (Male) | 1,765 | 50.43% |
| Gender | 1 (Female) | 1,735 | 49.57% |
| ProductType | 0 (Type A) | 2,454 | 70.11% |
| ProductType | 1 (Type B) | 1,046 | 29.89% |
| ChurnStatus | 0 (Not Churned) | 3,343 | 95.51% |
| ChurnStatus | 1 (Churned) | 157 | 4.49% |

### Correlation with ChurnStatus

| Feature | Correlation | Interpretation |
|---|---|---|
| Tenure | -0.2968 | Most predictive — longer tenure = lower churn risk |
| Income | -0.2856 | Higher income = lower churn risk |
| SupportCalls | +0.0262 | More calls = slightly higher churn risk |
| Age | -0.0021 | Not predictive |

---

## Key Findings

- **Tenure** and **Income** are the strongest negative predictors of churn — loyal, higher-income customers are least likely to leave.
- **SupportCalls** shows a small positive correlation with churn, suggesting service friction plays a role.
- **Age**, **Gender**, and **ProductType** show no meaningful relationship with churn.
- **High-risk profile:** customers with low tenure who generate a high volume of support calls.
- **Class imbalance:** only ~4.5% of records are churned — this must be addressed during model training (e.g., SMOTE, class weights).

---

## How to Run

### Requirements
- Python 3.8+
- Jupyter Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Steps
1. Place `customer_data.csv` in the same directory as `assianment1.ipynb`
2. Launch Jupyter: `jupyter notebook`
3. Open `assianment1.ipynb` and run all cells (**Kernel → Restart & Run All**)

---

## Project Files

| File | Description |
|---|---|
| `assianment1.ipynb` | Main Jupyter notebook with all code |
| `customer_data.csv` | Raw customer dataset (1,000 records) |
| `README.md` | This file |

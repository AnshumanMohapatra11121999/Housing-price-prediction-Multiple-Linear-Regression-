# 📊 Multiple Linear Regression — Housing Case Study

> **Author:** Anshuman Mohapatra
> **Dataset:** Delhi Housing Dataset (Area, Bedrooms, Bathrooms, Parking, etc. → Price)
> **📥 Download Dataset:** [https://drive.google.com/drive/folders/1qp2aQo-MON0jRS6c4Hvx3WFrkYSdPYiZ](https://drive.google.com/drive/folders/1qp2aQo-MON0jRS6c4Hvx3WFrkYSdPYiZ)
> **Tools Used:** Python · Pandas · NumPy · Matplotlib · Seaborn · Statsmodels · Scikit-learn
> **Project Type:** Supervised Learning — Multiple Linear Regression

---

## 📌 Table of Contents

1. [Objective](#objective)
2. [Dataset Overview](#dataset-overview)
3. [Step 1: Data Understanding & Cleaning](#step-1-data-understanding--cleaning)
4. [Step 2: Visualisation & EDA](#step-2-visualisation--eda)
5. [Step 3: Data Preparation](#step-3-data-preparation)
6. [Step 4: Splitting & Rescaling](#step-4-splitting--rescaling)
7. [Step 5: Building the Model](#step-5-building-the-model)
8. [Step 6: Residual Analysis](#step-6-residual-analysis)
9. [Step 7: Predictions on Test Set](#step-7-predictions-on-test-set)
10. [Key Concepts Learned](#key-concepts-learned)
11. [Conclusion & Takeaways](#conclusion--takeaways)

---

## 🎯 Objective

A real estate company has a dataset of property prices in the **Delhi region**. The company wants to:

1. Identify the **key variables** affecting house prices (e.g., area, bedrooms, parking, etc.)
2. Build a **linear model** that relates house prices quantitatively to these variables
3. Know the **accuracy** of the model — how well can these variables predict prices?

> **Key Insight:** Interpretation matters more than just prediction here. The company needs to understand *which* factors drive price and *by how much*.

---

## 📋 Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | Delhi Housing Dataset |
| **Shape** | 545 rows × 13 columns |
| **Target Variable** | `price` (INR) |
| **Missing Values** | None ✅ |

### Original Features

| Column | Type | Description |
|--------|------|-------------|
| `price` | int64 | Property price (Target) |
| `area` | int64 | Area in sq. ft. |
| `bedrooms` | int64 | Number of bedrooms |
| `bathrooms` | int64 | Number of bathrooms |
| `stories` | int64 | Number of floors/stories |
| `mainroad` | object (Yes/No) | Faces main road? |
| `guestroom` | object (Yes/No) | Has guestroom? |
| `basement` | object (Yes/No) | Has basement? |
| `hotwaterheating` | object (Yes/No) | Hot water heating? |
| `airconditioning` | object (Yes/No) | Air conditioning? |
| `parking` | int64 | Number of parking slots |
| `prefarea` | object (Yes/No) | In preferred area? |
| `furnishingstatus` | object | furnished / semi-furnished / unfurnished |

### Descriptive Statistics (Numeric Columns Only)

| Statistic | price | area | bedrooms | bathrooms | stories | parking |
|-----------|-------|------|----------|-----------|---------|---------|
| **Count** | 545 | 545 | 545 | 545 | 545 | 545 |
| **Mean** | 4,766,729 | 5150.5 | 2.97 | 1.29 | 1.81 | 0.69 |
| **Std** | 1,870,440 | 2170.1 | 0.74 | 0.50 | 0.87 | 0.86 |
| **Min** | 1,750,000 | 1650 | 1 | 1 | 1 | 0 |
| **Max** | 13,300,000 | 16200 | 6 | 4 | 4 | 3 |

---

## Step 1: Data Understanding & Cleaning

### Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
```

### Data Checks Performed

```python
housing.shape       # (545, 13)
housing.info()      # 6 int64, 7 object columns — No nulls
housing.describe()  # Descriptive statistics
```

> ✅ No missing values found. Dataset is clean and ready for preprocessing.

---

## Step 2: Visualisation & EDA

### 2.1 Visualising Numeric Variables

```python
sns.pairplot(housing)
plt.show()
```

**Key Observations:**
- `area` shows the strongest visual positive correlation with `price`
- `bedrooms` and `bathrooms` show mild positive trends
- No extreme outliers visible

### 2.2 Visualising Categorical Variables (Box Plots)

```python
plt.figure(figsize=(20, 12))
plt.subplot(2, 3, 1)
sns.boxplot(x='mainroad', y='price', data=housing)
```

**Key Observations:**
- Properties on the **main road** command significantly higher prices
- Houses with **air conditioning** have notably higher prices
- **Preferred area** houses are priced much higher
- Hot water heating and guestroom show smaller but visible premium effects

### 2.3 Correlation Heatmap

```python
sns.heatmap(housing.corr(), annot=True, cmap='YlGnBu')
```

| Feature | Correlation with Price |
|---------|------------------------|
| `area` | Moderate positive |
| `bathrooms` | Moderate positive |
| `stories` | Moderate positive |
| `airconditioning` | Moderate positive |
| `parking` | Weak positive |
| `bedrooms` | Weak positive |

---

## Step 3: Data Preparation

### 3.1 Converting Binary Categorical Columns (Yes/No → 1/0)

```python
varlist = ['mainroad', 'guestroom', 'basement',
           'hotwaterheating', 'airconditioning', 'prefarea']

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

housing[varlist] = housing[varlist].apply(binary_map)
```

### 3.2 Handling Multi-Level Categorical: `furnishingstatus`

```python
# Create dummy variables and drop first to avoid Dummy Variable Trap
status = pd.get_dummies(housing['furnishingstatus'], drop_first=True, dtype=int)

# Add to main dataframe and drop original column
housing = pd.concat([housing, status], axis=1)
housing.drop(['furnishingstatus'], axis=1, inplace=True)
```

**Encoding Logic after `drop_first=True`:**

| `semi-furnished` | `unfurnished` | Meaning |
|:---:|:---:|:---:|
| 0 | 0 | **furnished** (baseline/reference) |
| 1 | 0 | semi-furnished |
| 0 | 1 | unfurnished |

> **🚨 Dummy Variable Trap:** For `k` categories, always create `k-1` dummies.
> Including all `k` dummies causes **perfect multicollinearity** (columns always sum to 1).
> Fix: use `drop_first=True` in `pd.get_dummies()`.

---

## Step 4: Splitting & Rescaling

### 4.1 Train-Test Split (70/30)

```python
df_train, df_test = train_test_split(housing, train_size=0.7,
                                     test_size=0.3, random_state=100)
```

> **💡 Best Practice:** 70% training data, 30% testing. `random_state=100` ensures reproducibility.

| Split | Rows |
|-------|------|
| Training Set | ~381 rows |
| Test Set | ~164 rows |

### 4.2 Feature Scaling with MinMaxScaler

```python
scaler = MinMaxScaler()

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'parking', 'prefarea',
            'semi-furnished', 'unfurnished', 'price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
```

**MinMaxScaler Formula:**

$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

> **⚠️ Important:** Always call `fit_transform()` on training set only. Use `transform()` on test set.

---

## Step 5: Building the Model

### 5.1 Preparing X and y

```python
y_train = df_train.pop('price')
X_train = df_train
```

### 5.2 Building Model 1 (All Features)

```python
X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
print(lr_1.summary())
```

> **⚠️ Important:** `statsmodels` does NOT add an intercept by default. Always use `sm.add_constant()`.

### 5.3 Checking VIF (Variance Inflation Factor)

VIF measures how much a feature's variance is inflated due to multicollinearity with other features.

| VIF Value | Interpretation |
|-----------|---------------|
| 1 | No multicollinearity |
| 1–5 | Acceptable |
| 5–10 | Moderate — consider dropping |
| > 10 | High — must drop |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_calc(df):
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i)
                  for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by='VIF', ascending=False)
    return vif
```

### 5.4 Feature Selection Strategy

Iteratively drop features using **two criteria**:
1. **High VIF (> 5)** → multicollinearity
2. **High p-value (> 0.05)** → statistically insignificant

| Step | Feature Dropped | Reason |
|------|-----------------|--------|
| 1 | `bedrooms` | High p-value |
| 2 | `semi-furnished` | High p-value |
| 3 | `basement` | High p-value |
| ... | ... | ... |

> **🔑 Rule:** Drop only **ONE feature at a time** and re-check VIF + p-values after each drop.

### 5.5 Final Model Summary

**Features in Final Model:**

| Feature | Direction | P-value |
|---------|-----------|---------|
| `area` | Positive | < 0.05 ✅ |
| `bathrooms` | Positive | < 0.05 ✅ |
| `stories` | Positive | < 0.05 ✅ |
| `mainroad` | Positive | < 0.05 ✅ |
| `guestroom` | Positive | < 0.05 ✅ |
| `hotwaterheating` | Positive | < 0.05 ✅ |
| `airconditioning` | Positive | < 0.05 ✅ |
| `parking` | Positive | < 0.05 ✅ |
| `prefarea` | Positive | < 0.05 ✅ |
| `unfurnished` | Negative | < 0.05 ✅ |

**Final OLS Summary:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R-squared** | ~0.67 | 67% of price variance is explained |
| **Adj. R-squared** | ~0.66 | Penalized R² — still strong ✅ |
| **Prob (F-statistic)** | ~0.000 | Overall model is significant ✅ |
| **All p-values** | < 0.05 | All features are significant ✅ |

---

## Step 6: Residual Analysis

```python
y_train_price = lr_final.predict(X_train_lm_final)
res = y_train - y_train_price

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.distplot(res, ax=axes[0])
axes[1].scatter(y_train, res)
```

| Check | Expected | Result |
|-------|----------|--------|
| **Normality** | Bell-shaped distribution centred at 0 | ✅ Approximately normal |
| **Homoscedasticity** | Random scatter — no pattern | ✅ Checked |
| **Independence** | Durbin-Watson ≈ 2 | ✅ Passed |

---

## Step 7: Predictions on Test Set

```python
# Scale test data using the SAME scaler fitted on train (transform only)
df_test[num_vars] = scaler.transform(df_test[num_vars])

X_test = df_test[selected_features]
y_test = df_test['price']

X_test_lm = sm.add_constant(X_test)
y_pred = lr_final.predict(X_test_lm)
```

### Evaluation: Actual vs. Predicted Plot

```python
plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
```

### Test Set Performance

| Metric | Value |
|--------|-------|
| **R² (Train)** | ~0.671 |
| **R² (Test)** | ~0.668 |
| **Difference** | ~0.003 — minimal, no overfitting ✅ |

> **💡 Insight:** R² on train and test are nearly identical — model **generalizes well** to unseen data.

---

## 📚 Key Concepts Learned

### 1. What is Multiple Linear Regression?

$$\text{Price} = c + m_1 \cdot \text{area} + m_2 \cdot \text{bathrooms} + m_3 \cdot \text{stories} + \ldots$$

### 2. Dummy Variables & Dummy Variable Trap

| Concept | Detail |
|---------|--------|
| Purpose | Convert categorical features to numeric |
| Rule | For `k` categories → create `k-1` dummies |
| Trap | All `k` dummies → perfect multicollinearity |
| Fix | `drop_first=True` in `pd.get_dummies()` |

### 3. MinMaxScaler

| Aspect | Detail |
|--------|--------|
| Purpose | Scales features to [0, 1] range |
| Formula | `(X - X_min) / (X_max - X_min)` |
| Effect on R² | None — model performance unchanged |
| Fit rule | `fit_transform()` on train, `transform()` on test |

### 4. VIF (Variance Inflation Factor)

| VIF | Action |
|-----|--------|
| < 5 | Keep |
| 5–10 | Consider dropping |
| > 10 | Must drop |

### 5. Feature Selection Loop

```
While (any VIF > 5 OR any p-value > 0.05):
    Drop the feature with highest VIF or p-value
    Re-fit the model
    Re-check VIF + p-values
```

### 6. Residual Analysis Assumptions

| Assumption | Condition |
|-----------|-----------|
| Normality | Residuals normally distributed around 0 |
| Homoscedasticity | Constant variance of residuals |
| Independence | No autocorrelation (Durbin-Watson ≈ 2) |

---

## 🏁 Conclusion & Takeaways

### What I Learned

| # | Learning |
|---|---------|
| 1 | How to clean and prepare a real-world dataset for regression |
| 2 | Binary encoding (Yes/No → 1/0) using `.map()` |
| 3 | Dummy variable encoding using `pd.get_dummies(drop_first=True)` |
| 4 | The concept and danger of the **Dummy Variable Trap** |
| 5 | Feature scaling using **MinMaxScaler** |
| 6 | Iterative feature elimination using **VIF + p-values** |
| 7 | Using `sm.add_constant()` to include intercept in statsmodels |
| 8 | Interpreting the full OLS regression summary |
| 9 | Residual analysis to validate regression assumptions |
| 10 | Comparing Train R² and Test R² to detect overfitting |

### Final Model Summary

```
┌──────────────────────────────────────────────────────────────┐
│  Price = const + coeff₁·area + coeff₂·bathrooms             │
│        + coeff₃·stories + coeff₄·mainroad + ...             │
│                                                              │
│  R² (Train) ≈ 0.671   │   R² (Test) ≈ 0.668                │
│                                                              │
│  ✅ Model is statistically significant (Prob F ≈ 0.000)      │
│  ✅ All features have p-values < 0.05                        │
│  ✅ All VIFs are within acceptable range (< 5)               │
│  ✅ Residuals are approximately normal                       │
│  ✅ Model generalizes well (Train ≈ Test R²)                 │
└──────────────────────────────────────────────────────────────┘
```

### MLR vs SLR — Key Differences

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|--------|--------------------------|---------------------------|
| Predictors | 1 | Multiple |
| R² vs Adj R² | Both usually same | Always use Adj R² for comparison |
| Multicollinearity | Not a concern | Must check with VIF |
| Feature Selection | Not needed | Iterative elimination required |
| Dummy Variables | Usually not needed | Critical for categorical data |

---

> *These notes were prepared as a learning showcase for the Multiple Linear Regression — Housing Case Study in Python.*
> *Based on real notebook work with the Delhi Housing Dataset.*

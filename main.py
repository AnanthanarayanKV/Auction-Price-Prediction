import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

from cleaning import dataCleaning

# ── 1. Loading cleaned data ────────────────────────────────────────
path = Path("cleaned_data.csv")
if path.is_file():
    print("Cleaned data found – loading.")
else:
    print("Cleaned data not found – running cleaner…")
    dataCleaning()

df = pd.read_csv('cleaned_data.csv')

# ── 2. Removing price outliers using IQR method ─────────────────────────────────────
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

print(f"\nRows before outlier removal: {len(df):,}")

IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)].copy()

print(f"\nRows after outlier removal: {len(df):,}")

# ── 3. Defining feature groups ───────────────────────────────────────────────────
# String columns that will be mean-encoded in step 5
string_cols = ['make', 'model', 'trim', 'full_spec']

# Columns to exclude entirely from X
non_feature_cols = {'price', 'year'} | set(string_cols) #this contains a list of attributes which doesn't need to be trained
feature_cols = []

for c in df.columns:
    if c not in non_feature_cols:
        feature_cols.append(c)

# Columns to standardise (continuous numeric, not binary OHE flags)
cols_to_scale = ['mileage', 'mileage_per_year', 'vehicle_age',
                 'engine_hp', 'brand_popularity', 'owner_count']
# Filter to only those that actually exist after OHE
cols_to_scale = [c for c in cols_to_scale if c in feature_cols]

# ── 4. Train / test split FIRST (before any fit / encoding / scaling) ─────────
y = df['price']
X_str = df[string_cols]          # raw string cols, encoded later
X_num = df[feature_cols]         # numeric + OHE boolean cols

X_train_num, X_test_num,X_train_str, X_test_str, y_train, y_test = train_test_split(X_num, X_str, y, test_size=0.2, random_state=42)

# ── 5. Mean encoding – fitted on training fold only ────────────────────────────
def mean_encode(train_X, test_X, train_y, col, new_col):
    means       = train_y.groupby(train_X[col]).mean()
    global_mean = train_y.mean()   # fallback for unseen categories
    train_X[new_col] = train_X[col].map(means).fillna(global_mean)
    test_X[new_col]  = test_X[col].map(means).fillna(global_mean)

mean_encode(X_train_str, X_test_str, y_train, 'full_spec', 'spec_encoded')
mean_encode(X_train_str, X_test_str, y_train, 'model',     'model_encoded')
mean_encode(X_train_str, X_test_str, y_train, 'make',      'make_encoded')

# Attach mean-encoded columns to the numeric feature frames
for col in ['spec_encoded', 'model_encoded', 'make_encoded']:
    X_train_num[col] = X_train_str[col].values
    X_test_num[col]  = X_test_str[col].values

# ── 6. Feature scaling – fitted on training fold only ────────────────────────
scaler = StandardScaler()
X_train_num[cols_to_scale] = scaler.fit_transform(X_train_num[cols_to_scale])
X_test_num[cols_to_scale]  = scaler.transform(X_test_num[cols_to_scale])

X_train = X_train_num
X_test  = X_test_num

print(f"\nFinal feature count : {X_train.shape[1]}")
print(f"Training samples    : {len(X_train):,}")
print(f"Test samples        : {len(X_test):,}")

# ── 7. Correlation heatmap (training set only) ─────────────────────────────────
print("\nSaving correlation heatmap…")
train_plot = X_train.copy()
train_plot['price'] = y_train.values

sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(28, 22))
sns.heatmap(
    train_plot.corr(numeric_only=True),
    annot=True, cmap='coolwarm', fmt='.2f',
    annot_kws={"size": 8}, linewidths=0.5
)
plt.title("Feature Correlation Heatmap (training set)", fontsize=22)
plt.xticks(rotation=45, ha='right')
plt.savefig('large_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Heatmap saved to large_heatmap.png")

# ── 8. Model – Random Forest ────────────────────────────────
y_train_log = np.log1p(y_train)
rf_model = RandomForestRegressor(n_estimators = 100,max_depth = 20,min_samples_leaf= 5,n_jobs = -1,random_state = 42)
rf_model.fit(X_train, y_train_log)
y_pred_log = rf_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)

 
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
 
print(f"\n{'─'*40}")
print(f"  Random Forest R²   : {r2:.4f}")
print(f"  Random Forest MAE  : ${mae:,.2f}")
print(f"  Random Forest MAPE : {mape*100:.2f}%")
print(f"{'─'*40}")
print(f"  Interpretation: predictions are off by ~{mape*100:.1f}% on average")

# ── 9. Residuals plot ──────────────────────────────────────────────────────────
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.15, color='steelblue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual ($)")
plt.title("Polynomial Regression – Residuals")
plt.savefig('poly_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Residuals plot saved to poly_residuals.png")

# ── 10. Depreciation rate per brand ───────────────────────────────────────────
# Uses the original un-scaled df so vehicle_age is interpretable
depreciation_rates = {}
for brand in df['make'].unique():
    subset = df[df['make'] == brand]
    if len(subset) > 10:
        depreciation_rates[brand] = subset['price'].corr(subset['vehicle_age'])

sorted_dep = pd.Series(depreciation_rates).sort_values()
print("\nHighest depreciation (price drops fastest with age):")
print(sorted_dep.head(5).to_string())
print("\nLowest depreciation  (value holds best):")
print(sorted_dep.tail(5).to_string())
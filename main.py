from sklearn.model_selection import train_test_split
import pandas as pd
from cleaning import dataCleaning
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

scale = StandardScaler()

cwd = Path.cwd()
path = Path(f"{cwd}/cleaned_data.csv")

if path.is_file():
    print("Cleaned data available and ready to use")
    
else:
    print("Cleaned data not found,\ncleaning data....\n")
    dataCleaning()
    
df = pd.read_csv('cleaned_data.csv')

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

clean_df = df[(df['price']>=lower_limit) & (df['price']<=upper_limit) ]
data_to_scale = ['mileage','mileage_per_year','vehicle_age','engine_hp']
clean_df[data_to_scale] = scale.fit_transform(clean_df[data_to_scale])

# print(clean_df.columns)
clean_df.drop(columns=['model', 'trim'], inplace=True)
# print(clean_df.head())

sns.set_context("paper", font_scale=1.5)

# # 2. Define a much larger figure size (Width, Height in inches)
plt.figure(figsize=(25, 20))

# # 3. Increase annotation size using annot_kws
# # 'fmt' determines decimal places, 'annot_kws' sets font properties
sns.heatmap(clean_df.corr(numeric_only=True), 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            annot_kws={"size": 12},  # Size of numbers inside the boxes
            linewidths=1)          # Adds spacing between elements for clarit
plt.title("Enhanced Feature Correlation Heatmap", fontsize=25)
plt.xticks(rotation=45, ha='right')  # Tilts x-axis labels for better fit

# # Since you mentioned the FigureCanvasAgg warning earlier:
plt.savefig('large_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as large_heatmap.png")

feature = [
    'vehicle_age',
    'mileage',
    'engine_hp',
    'owner_count',
    'mileage_per_year',
    'spec_encoded',
    'model_encoded'
]

X = clean_df[feature]
y = clean_df['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# 2. Train the model
poly_model.fit(X_train, y_train)

# 3. Predict
y_poly_pred = poly_model.predict(X_test)

print(f"Polynomial R2 Score: {r2_score(y_test, y_poly_pred):.4f}")
print(f"Polynomial MAE: ${mean_absolute_error(y_test, y_poly_pred):.2f}")

# Quick check for the new residuals
poly_residuals = y_test - y_poly_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_poly_pred, poly_residuals, alpha=0.1, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Polynomial Residuals: Much Flatter!')
plt.savefig('poly_residuals.png')


# Calculate the average price per brand for 'New' vs 'Old' cars
brand_stats = clean_df.groupby('make').agg(
    avg_price=('price', 'mean'),
    avg_age=('vehicle_age', 'mean')
)

# Calculate Depreciation Rate (Price drop per year of age)
# We can use a simple correlation for each brand group
depreciation_rates = {}
for brand in clean_df['make'].unique():
    subset = clean_df[clean_df['make'] == brand]
    # Correlation between age and price (should be negative)
    corr = subset['price'].corr(subset['vehicle_age'])
    depreciation_rates[brand] = corr

# Sort: More negative = Higher Depreciation
sorted_depreciation = pd.Series(depreciation_rates).sort_values()
print("Highest Depreciation (Value drops fastest):", sorted_depreciation.head(3))
print("Lowest Depreciation (Value holds best):", sorted_depreciation.tail(3))
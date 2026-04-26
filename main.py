import pandas as pd
from cleaning import dataCleaning
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

print(f"Lower limit is {lower_limit} and upper limit is {upper_limit}\n")

clean_df = df[(df['price']>=lower_limit) & (df['price']<=upper_limit) ]
print(f"Original rows = {len(df)}\n")
print(f"Cleaned rows = {len(clean_df)}\n")
print(f"Outliers removed = {len(df) - len(clean_df)}")

data_to_scale = ['mileage','mileage_per_year','vehicle_age','engine_hp']

clean_df[data_to_scale] = scale.fit_transform(clean_df[data_to_scale])

print(clean_df.head())

corr_matrix = clean_df.corr()

# Plot it
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation with Price")
plt.savefig('correlation.png')

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

print(clean_df.columns)
clean_df.drop(columns=['make', 'model', 'trim'], inplace=True)
print(clean_df.head())

sns.set_context("paper", font_scale=1.5)

# 2. Define a much larger figure size (Width, Height in inches)
plt.figure(figsize=(25, 20))

# 3. Increase annotation size using annot_kws
# 'fmt' determines decimal places, 'annot_kws' sets font properties
sns.heatmap(clean_df.corr(numeric_only=True), 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            annot_kws={"size": 12},  # Size of numbers inside the boxes
            linewidths=1)          # Adds spacing between elements for clarity

plt.title("Enhanced Feature Correlation Heatmap", fontsize=25)
plt.xticks(rotation=45, ha='right')  # Tilts x-axis labels for better fit

# Since you mentioned the FigureCanvasAgg warning earlier:
plt.savefig('large_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as large_heatmap.png")
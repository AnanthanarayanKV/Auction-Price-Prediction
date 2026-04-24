import pandas as pd
from sklearn.preprocessing import LabelEncoder

OHE = LabelEncoder()

df = pd.read_csv("vehicle_price_prediction.csv")

missing_count = df.isnull().sum()
total_missing = missing_count.sum()

if total_missing > 0:
    print(f"Total missing values: {total_missing}")
    print("Breakdown by column:")
    print(missing_count[missing_count > 0]) 
    error_percent =  (total_missing/len(df))*100
    print(f"Missing percentage is {total_missing}/{len(df)} = {error_percent}")
else:
    print("No missing values detected.")

drivetrain_map = {
    'FWD': 0,
    'RWD': 1,
    'AWD': 2
}

transmission_map = {
    'Manual':0,
    'Automatic': 1
}

fuel_map = {
    'Diesel': 0,
    'Gasoline': 1,
    'Electric': 2
}

df['drivetrain'] = df['drivetrain'].map(drivetrain_map)
df['transmission'] = df['transmission'].map(transmission_map)
df['fuel_type'] = df['fuel_type'].map(fuel_map)

print("\n--- Statistical Summary ---")
print(df.describe())
print("\n--- First 5 Rows ---")
print(df.head())
print(df.tail())
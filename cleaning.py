import pandas as pd


def dataCleaning():
    # ── Load raw data ──────────────────────────────────────────────────────────
    df = pd.read_csv("car_auction_data.csv")
    print("\nPre Cleaning – missing values:\n", df.isnull().sum().sum())

    # ── Missing values ─────────────────────────────────────────────────────────
    # None in accident_history means no prior accident – fill is more useful
    df['accident_history'] = df['accident_history'].fillna("No_Accident")
    df['price'] = df['price'].fillna(df['price'].mean())
    df['mileage'] = df['mileage'].fillna(df['mileage'].mean())
    df['mileage_per_year'] = df['mileage_per_year'].fillna(df['mileage_per_year'].mean())
    df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].mean())
    
    # ── Ordinal encoding ───────────────────────────────────────────────────────
    condition_map = {'Excellent': 2, 'Good': 1, 'Fair': 0}
    accident_map  = {'No_Accident': 2, 'Minor': 1, 'Major': 0}

    df['condition']        = df['condition'].map(condition_map)
    df['accident_history'] = df['accident_history'].map(accident_map)

    # ── Composite spec column ──────────────────────────────────────────────────
    df['full_spec'] = df['make'] + "_" + df['model'] + "_" + df['trim']
    spec_means = df.groupby('full_spec')['price'].mean()
    df['spec_encoded'] = df['full_spec'].map(spec_means)

    # ── One-hot encoding ───────────────────────────────────────────────────────
    ohe_cols = [
        'transmission',
        'fuel_type', 'drivetrain', 'body_type',
        'seller_type', 'exterior_color', 'interior_color'
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # ── 'make', 'model', 'trim', 'full_spec' kept as raw strings ──────────────

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv('cleaned_data.csv', index=False)
    print("\nCleaning complete. Columns:\n", list(df.columns))
    print("\nPost Cleaning – missing values:\n", df.isnull().sum().sum())


if __name__ == '__main__':
    dataCleaning()
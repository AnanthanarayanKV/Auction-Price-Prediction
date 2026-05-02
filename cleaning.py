import pandas as pd


def dataCleaning():
    # ── Load raw data ──────────────────────────────────────────────────────────
    df = pd.read_csv("vehicle_price_prediction.csv")
    print("\nPre Cleaning – missing values:\n", df.isnull().sum())

    # ── Missing values ─────────────────────────────────────────────────────────
    # None in accident_history means no prior accident – fill is more useful
    df['accident_history'] = df['accident_history'].fillna("No_Accident")

    # ── Ordinal encoding ───────────────────────────────────────────────────────
    condition_map = {'Excellent': 2, 'Good': 1, 'Fair': 0}
    accident_map  = {'No_Accident': 2, 'Minor': 1, 'Major': 0}

    df['condition']        = df['condition'].map(condition_map)
    df['accident_history'] = df['accident_history'].map(accident_map)

    # ── Composite spec column ──────────────────────────────────────────────────
    df['full_spec'] = df['make'] + "_" + df['model'] + "_" + df['trim']

    # ── One-hot encoding ───────────────────────────────────────────────────────
    # transmission: typically 2-3 values (Manual / Automatic / CVT) – OHE fine
    # Colors are already simplified to ~8-12 values – OHE fine
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


if __name__ == '__main__':
    dataCleaning()
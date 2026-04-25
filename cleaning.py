import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

OHE = OneHotEncoder()

def dataCleaning():
    #Loading my data into python
    df = pd.read_csv("vehicle_price_prediction.csv")

    print("\nPre Cleaning",df.isnull().sum())
    #there are no missing values except None in accident_history.
    #In this context None is a valid value implying there has been no accident previously.

    df['accident_history'] = df['accident_history'].fillna("No_Accident")

    #mapping the needed columns to be encoded
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

    condition_map = {
        'Excellent': 2,
        'Good': 1,
        'Fair': 0
    }

    accident_map = {
        'No_Accident':2,
        'Minor':1,
        'Major':0
    }

    #encoding the columns in dataset
    df['drivetrain'] = df['drivetrain'].map(drivetrain_map)
    df['transmission'] = df['transmission'].map(transmission_map)
    df['fuel_type'] = df['fuel_type'].map(fuel_map)
    df['condition'] = df['condition'].map(condition_map)
    df['accident_history'] = df['accident_history'].map(accident_map)

    unique_seller = df['seller_type'].unique()

    seller_transform = ColumnTransformer(
        [("encoded", OHE, ['seller_type'])], 
        remainder='passthrough',
        verbose_feature_names_out=False
    )


    df_encoded = seller_transform.fit_transform(df)
    column_data = seller_transform.get_feature_names_out()
    df_final = pd.DataFrame(df_encoded, columns=column_data)

    # print("\n--- First 5 Rows ---")
    # print(df_final.head())
    # print(df_final.tail())

    df_final.to_csv('cleaned_data.csv')

    #print("\nPost cleaning",df_final.isnull().sum())
    
    
if __name__ == '__main__':
    dataCleaning()
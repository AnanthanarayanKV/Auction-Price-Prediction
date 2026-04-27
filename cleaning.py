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
    
    df['full_spec'] = df['make'] + "_" + df['model'] + "_" + df['trim']

    spec_means = df.groupby('full_spec')['price'].mean()
    df['spec_encoded'] = df['full_spec'].map(spec_means)
    
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
    
    # Convert Fuel, Drive-train, and Body Type at once
    df= pd.get_dummies(df, columns=['fuel_type', 'drivetrain', 'body_type','seller_type','exterior_color','interior_color'],drop_first=True)
    
    df['condition'] = df['condition'].map(condition_map)
    df['accident_history'] = df['accident_history'].map(accident_map)
    model_means = df.groupby('model')['price'].mean()
    df['model_encoded'] = df['model'].map(model_means)
    print(df.columns)
    df.drop(['fuel_type', 'drivetrain', 'body_type', 'seller_type', 'full_spec'], axis=1, errors='ignore', inplace=True)
    print(df.columns)
    
    # print("\n--- First 5 Rows ---")
    # print(df_final.head())
    # print(df_final.tail())

    df.to_csv('cleaned_data.csv')

    #print("\nPost cleaning",df_final.isnull().sum())
    
    
if __name__ == '__main__':
    dataCleaning()
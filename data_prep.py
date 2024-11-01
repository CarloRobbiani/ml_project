import requests
import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_data():
    # Define your variables
    vars = ['AGEP', 'SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'WAGP']  # Example variables

    base_url = "https://api.census.gov/data/2022/acs/acs5/pums"
    variables = ",".join(vars)

    predicates = {
        "get": variables,
        "for": "state:01",  # Example: state Alabama (FIPS code 01)
        "key": "b75e9580f6d16319ae86b81a6466dd818cc03119"
    }

    df = None

    # Make the API request
    response = requests.get(base_url, params=predicates)

    if response.headers.get('Content-Type') == 'application/json;charset=utf-8':
        data = response.json()

        # The first row contains the column names, and the remaining rows contain the data
        headers = data[0]  # First row (column names)
        rows = data[1:]    # Remaining rows (data)

        # Load into a pandas DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Display the DataFrame
        return df
    else:
        print("Unexpected content type:", response.headers.get('Content-Type'))


def prepare_data(df):
    df['AGEP'] = pd.to_numeric(df['AGEP'], errors='coerce').astype('Int64')

    # Sex (SEX), Race (RAC2P), Hispanic Origin (HISP), Educational Attainment (SCHL),
    # and Employment Status (ESR) are likely categorical
    categorical_columns = ['SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Wages (WAGP) should be numeric
    df['WAGP'] = pd.to_numeric(df['WAGP'], errors='coerce')

    df.dtypes
    #Removing Salarys below 4
    df = df[df["WAGP"] > 4]


    return df

def select_features(df):
    # Select relevant features for wage prediction
    features = ['AGEP', 'SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR']  # Example features
    X = df[features]
    y = df['WAGP']  # Target variable
    y = df["WAGP"].values.reshape(-1, 1)
    y = preprocessing.normalize(y, axis=0)
    y = y.flatten()

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, columns=['SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR'])
    return X,y

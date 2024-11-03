import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


def load_data():
    """
    Loads the ACS PUMS data either from a local file or via API request.
    Always includes 'OCCP' in the variables.

    Returns:
        df (pd.DataFrame): The loaded data as a pandas DataFrame.
    """
    # Define the path where you want to save your data
    data_file = 'data.csv'

    if os.path.exists(data_file):
        # If the data file exists, load the data from the file
        print("Loading data from local file...")
        df = pd.read_csv(data_file)
    else:
        # If the data file does not exist, load the data from the API
        print("Making API request...")

        # Define your variables, always including 'OCCP'
        vars = ['AGEP', 'SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'WAGP', 'WKHP', 'OCCP']

        base_url = "https://api.census.gov/data/2022/acs/acs5/pums"
        variables = ",".join(vars)

        api_key = 'b75e9580f6d16319ae86b81a6466dd818cc03119'

        predicates = {
            "get": variables,
            "for": "state:*",  # Fetch data for all states
            "key": api_key
        }

        # Make the API request
        response = requests.get(base_url, params=predicates)
        print(f"Response Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()

                # The first row contains the column names, and the remaining rows contain the data
                headers = data[0]  # First row (column names)
                rows = data[1:]  # Remaining rows (data)

                # Load into a pandas DataFrame
                df = pd.DataFrame(rows, columns=headers)

                # Save the DataFrame to a CSV file for future use
                df.to_csv(data_file, index=False)
                print("Data saved to local file.")
            except ValueError as e:
                print("JSON decoding failed:", e)
                print("Response Content:")
                print(response.text)
                return None
        else:
            print("Failed to fetch data from the API.")
            print(f"Status Code: {response.status_code}")
            print("Response Content:")
            print(response.text)
            return None

        exit("Data loaded successfully.")

    return df


def prepare_data(df, specific_occupation=False):
    """
    Prepares the data by converting data types, filtering, and transforming variables.
    Decides whether to filter by specific occupations.

    Args:
        df (pd.DataFrame): The raw data DataFrame.
        specific_occupation (bool): Whether to filter by specific occupations.

    Returns:
        df (pd.DataFrame): The prepared data DataFrame.
    """
    # Convert data types
    df['AGEP'] = pd.to_numeric(df['AGEP'], errors='coerce').astype('Int64')
    df['WKHP'] = pd.to_numeric(df['WKHP'], errors='coerce').astype('Int64')  # Hours worked per week
    df['OCCP'] = df['OCCP'].astype('category')  # Occupation code

    # Filter based on hours worked
    df = df[df['WKHP'] >= 40]  # Keep only individuals who work at least 40 hours per week

    # Filter by specific occupations if needed
    if specific_occupation:
        # Replace '2310' and '2320' with the occupation codes you're interested in
        occupation_codes = ['2310', '2320']  # Example occupation codes
        df = df[df['OCCP'].isin(occupation_codes)]
        print(f"Data filtered for specific occupations: {df['OCCP'].unique()}")

    # Convert categorical columns to 'category' dtype
    categorical_columns = ['SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'OCCP']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Wages (WAGP) should be numeric
    df['WAGP'] = pd.to_numeric(df['WAGP'], errors='coerce')

    # Remove records with non-positive wages
    df = df[df['WAGP'] > 0]

    # Apply logarithmic transformation to 'WAGP'
    df['WAGP_LOG'] = np.log1p(df['WAGP'])

    return df


def select_features(df, specific_occupation=False):
    """
    Selects features and the target variable for modeling.
    Decides whether to include 'OCCP' in features.

    Args:
        df (pd.DataFrame): The prepared data DataFrame.
        specific_occupation (bool): Whether to include 'OCCP' in features.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    """
    # Select relevant features for wage prediction
    features = ['AGEP', 'SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'WKHP']
    categorical_columns = ['SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR']

    if not specific_occupation:
        # Include 'OCCP' in features when not filtering by specific occupations
        features.append('OCCP')
        categorical_columns.append('OCCP')

    X = df[features]
    y = df['WAGP_LOG']  # Use the log-transformed target variable

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, columns=categorical_columns)

    return X, y


def prepare_for_model(X_train, X_test, y_train, y_test):
    """
    Scales features and converts data to PyTorch tensors.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        train_loader (DataLoader): DataLoader for training data.
        X_train_tensor (Tensor): Training features as tensor.
        y_train_tensor (Tensor): Training targets as tensor.
        X_test_tensor (Tensor): Testing features as tensor.
        y_test_tensor (Tensor): Testing targets as tensor.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return X_train_scaled, train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
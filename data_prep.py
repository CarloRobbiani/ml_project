import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Loads the ACS PUMS data either from a local file or via API request.
    Always includes 'OCCP' and ancestry variables.

    Returns:
        df (pd.DataFrame): The loaded data as a pandas DataFrame.
    """
    # Define the path where you want to save your data
    reload_data = False

    data_file = 'data.csv'

    if os.path.exists(data_file) and not reload_data:
        print("Loading data from local file...")
        df = pd.read_csv(data_file)
    else:
        print("Making API request...")

        # Define your variables, always including 'OCCP', 'ANC1P', 'ANC2P'
        vars = ['AGEP', 'SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'WAGP', 'WKHP', 'OCCP', 'ANC1P', 'ANC2P']
        base_url = "https://api.census.gov/data/2022/acs/acs5/pums"
        variables = ",".join(vars)

        # Retrieve API key from environment variable
        api_key = 'b75e9580f6d16319ae86b81a6466dd818cc03119'

        predicates = {
            "get": variables,
            "for": "state:*",  # Fetch data for all states
            "key": api_key
        }

        try:
            response = requests.get(base_url, params=predicates)
            response.raise_for_status()
            data = response.json()

            headers = data[0]
            rows = data[1:]

            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(data_file, index=False)
            print("Data saved to local file.")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except ValueError as e:
            print("JSON decoding failed:", e)
            print("Response Content:")
            print(response.text)
            return None
        exit("Data loaded")
    return df

# Initialize a label cache to minimize API calls
label_cache_file = 'label_cache.pkl'

# Load existing cache if available
if os.path.exists(label_cache_file):
    with open(label_cache_file, 'rb') as f:
        label_cache = pickle.load(f)
else:
    label_cache = {}

def get_value_labels_from_api(variable_name):
    """
    Fetches code-to-label mapping for a given variable from the Census API.
    Utilizes caching to minimize API calls.

    Args:
        variable_name (str): The variable name (e.g., 'RAC2P').

    Returns:
        dict: Mapping of codes to labels.
    """
    if variable_name in label_cache:
        return label_cache[variable_name]

    base_url = f"https://api.census.gov/data/2022/acs/acs5/pums/variables/{variable_name}.json"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        variable_metadata = response.json()

        if 'values' in variable_metadata:
            value_labels = variable_metadata['values']['item']
            label_cache[variable_name] = value_labels
            return value_labels
        else:
            print(f"No value labels found for variable '{variable_name}'.")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch metadata for variable '{variable_name}': {e}")
        return {}

def save_label_cache():
    """
    Saves the label cache to a file for future use.
    """
    with open(label_cache_file, 'wb') as f:
        pickle.dump(label_cache, f)


def visualize_wage_distribution(df, save_fig=False):
    """
    Visualizes the distribution of wages using a histogram and boxplot.

    Args:
        df (pd.DataFrame): The DataFrame containing 'WAGP'.
        save_fig (bool): Whether to save the figures as PNG files.
    """
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['WAGP'], bins=50, kde=True)
    plt.title('Wage Distribution')
    plt.xlabel('Wage')
    plt.ylabel('Frequency')
    if save_fig:
        plt.savefig('wage_distribution_histogram.png')
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['WAGP'])
    plt.title('Wage Distribution Boxplot')
    plt.xlabel('Wage')
    if save_fig:
        plt.savefig('wage_distribution_boxplot.png')
    plt.show()


def identify_outliers_iqr(df, column='WAGP'):
    """
    Identifies outliers in a column using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to analyze.

    Returns:
        lower_bound (float): The lower bound for non-outlier data.
        upper_bound (float): The upper bound for non-outlier data.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"{column} - Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"{column} - Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    return lower_bound, upper_bound


def remove_outliers(df, column='WAGP'):
    """
    Removes outliers from the DataFrame based on IQR.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to analyze.

    Returns:
        df_filtered (pd.DataFrame): The DataFrame without outliers.
    """
    lower_bound, upper_bound = identify_outliers_iqr(df, column)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    print(f"Removed {df.shape[0] - df_filtered.shape[0]} outliers from {column}.")
    return df_filtered


def cap_outliers(df, column='WAGP'):
    """
    Caps outliers in the DataFrame based on IQR.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to analyze.

    Returns:
        df_capped (pd.DataFrame): The DataFrame with outliers capped.
    """
    lower_bound, upper_bound = identify_outliers_iqr(df, column)
    df_capped = df.copy()
    num_capped_upper = (df_capped[column] > upper_bound).sum()
    df_capped[column] = np.where(df_capped[column] > upper_bound, upper_bound, df_capped[column])
    print(f"Capped {num_capped_upper} outliers in {column} to the upper bound of {upper_bound}.")
    return df_capped


def prepare_data(df, specific_occupation=False, handle_outliers=False, method='remove'):
    """
    Prepares the data by converting data types, filtering, transforming variables,
    mapping codes to labels, and handling outliers.

    Args:
        df (pd.DataFrame): The raw data DataFrame.
        specific_occupation (bool): Whether to filter by specific occupations.
        handle_outliers (bool): Whether to handle outliers.
        method (str): The method to handle outliers ('remove' or 'cap').

    Returns:
        df (pd.DataFrame): The prepared data DataFrame with label columns added.
    """
    # Convert data types
    df['AGEP'] = pd.to_numeric(df['AGEP'], errors='coerce').astype('Int64')
    df['WKHP'] = pd.to_numeric(df['WKHP'], errors='coerce').astype('Int64')  # Hours worked per week
    df['OCCP'] = df['OCCP'].astype(str)  # Ensure OCCP codes are strings
    df['ANC1P'] = df['ANC1P'].astype(str)
    df['ANC2P'] = df['ANC2P'].astype(str)

    # Filter based on hours worked
    df = df[(df['WKHP'] >= 35) & (df['WKHP'] <= 65)].copy()  # Full-time workers

    # Filter by specific occupations if needed
    it_occ_codes = [
        '0110', '1005', '1006', '1007', '1010', '1021', '1022', '1031', '1032',
        '1050', '1065', '1105', '1106', '1108'
    ]

    if specific_occupation:
        df = df[df['OCCP'].isin(it_occ_codes)].copy()
        print(f"Data filtered for specific occupations: {df['OCCP'].unique()}")

    # Convert categorical columns to 'category' dtype
    categorical_columns = ['SEX', 'RAC2P', 'HISP', 'SCHL', 'ESR', 'OCCP', 'ANC1P', 'ANC2P']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Wages (WAGP) should be numeric
    df['WAGP'] = pd.to_numeric(df['WAGP'], errors='coerce')

    # Remove records with non-positive wages
    df = df[df['WAGP'] > 0].copy()

    # Keep only the 8 races with the most data in 'RAC2P'
    top_8_races = df['RAC2P'].value_counts().nlargest(8).index.tolist()
    df = df[df['RAC2P'].isin(top_8_races)].copy()
    print(f"Top 8 races kept in 'RAC2P': {top_8_races}")

    # Remove unused categories in 'RAC2P'
    df['RAC2P'] = df['RAC2P'].cat.remove_unused_categories()

    # Map codes to labels using the Census API
    code_variables = ['RAC2P', 'SCHL', 'SEX', 'HISP', 'ESR', 'OCCP', 'ANC1P', 'ANC2P']
    for var in code_variables:
        print(f"Fetching labels for variable '{var}'...")
        labels_dict = get_value_labels_from_api(var)
        if labels_dict:
            # Determine the maximum key length for padding
            key_length = max(len(k) for k in labels_dict.keys())
            # Pad the data codes with leading zeros to match label keys
            padded_codes = df[var].astype(str).str.zfill(key_length)
            # Map the padded codes to labels
            mapped = padded_codes.map(labels_dict)
            # Fill NaN values with the original padded codes (as strings)
            mapped_filled = mapped.fillna(padded_codes)
            # Assign to the new label column as 'object' dtype
            df[f'{var}_LABEL'] = mapped_filled.astype('object')
        else:
            # If no labels found, keep the original codes as object type
            df[f'{var}_LABEL'] = df[var].astype('object')

    # Visualize Wage Distribution
    print("\nVisualizing Wage Distribution Before Handling Outliers:")
    visualize_wage_distribution(df, save_fig=False)

    # Handle outliers if specified
    if handle_outliers:
        print("\nHandling outliers in 'WAGP'...")
        if method == 'remove':
            df = remove_outliers(df, column='WAGP')
        elif method == 'cap':
            df = cap_outliers(df, column='WAGP')
        else:
            raise ValueError("Invalid method for handling outliers. Choose 'remove' or 'cap'.")

        # Visualize Wage Distribution After Handling Outliers
        print("\nVisualizing Wage Distribution After Handling Outliers:")
        visualize_wage_distribution(df, save_fig=False)

    # Save the updated label cache
    save_label_cache()

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
    # Select relevant features for wage prediction using label columns
    features = ['AGEP', 'SEX_LABEL', 'RAC2P_LABEL', 'HISP_LABEL', 'SCHL_LABEL', 'ESR_LABEL', 'WKHP', 'ANC1P_LABEL', 'ANC2P_LABEL']
    categorical_columns = ['SEX_LABEL', 'RAC2P_LABEL', 'HISP_LABEL', 'SCHL_LABEL', 'ESR_LABEL', 'ANC1P_LABEL', 'ANC2P_LABEL']

    if not specific_occupation:
        features.append('OCCP_LABEL')
        categorical_columns.append('OCCP_LABEL')

    X = df[features].copy()
    y = df['WAGP']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

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

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from UCI_train import train_UCI
from UCI_plots import *
  

def load_UCI_api():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 

    # Combine features and target
    data = pd.concat([X, y], axis=1)

    # Handle missing values
    data = data.replace(' ?', np.nan).dropna()

    # Split features and target
    X = data.drop('income', axis=1)
    y = data['income']
    return X,y
def load_UCI_local():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    data = pd.read_csv("UCI_data/adult.data", header=None, names=column_names, sep=', ', engine='python')
    test_data = pd.read_csv("UCI_data/adult.test", header=None, names=column_names, sep=', ', engine='python')
    
    # Replace '?' with NaN and fill with mode
    data = data.replace(' ?', np.nan).dropna()
    test_data = test_data.replace(' ?', np.nan).dropna()
    """
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
        """
    
    # Encode target variable
    def encode_income(x):
        if x.strip() in ['<=50K', '<=50K.']:
            return 0
        elif x.strip() in ['>50K', '>50K.']:
            return 1
        else:
            return np.nan

    data['income'] = data['income'].apply(encode_income)
    test_data['income'] = test_data['income'].apply(encode_income)
    
    X_train = data.drop('income', axis=1)
    X_test = test_data.drop('income', axis=1)
    y_train = data['income']
    y_test = test_data['income']
    
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Finished loading")
    return X_train, X_test, y_train, y_test, preprocessor

X_train,X_test,y_train,y_test, preprocessor =load_UCI_local()

y_pred, clf = train_UCI(preprocessor, X_train, y_train, X_test, y_test)

conf_matrix(y_test, y_pred)

#feature_importance(numeric_features, clf, categorical_features)

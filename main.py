from data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from models import RegressionModel  # Ensure this contains your RegressionModel definition
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

def GradBoost(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Regressor and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target variable.
    """
    # Initialize the model
    model = GradientBoostingRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f'Gradient Boosting Regressor - Mean Squared Error: {mse:.2f}')

    # Optionally, plot feature importances
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.sort_values(ascending=False).head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def own_model(train_loader, X_train_scaled, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    """
    Trains a custom neural network model for regression and plots learning curves.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        X_train_scaled (np.ndarray): Scaled training features.
        X_train_tensor (Tensor): Training features as tensor.
        y_train_tensor (Tensor): Training targets as tensor.
        X_test_tensor (Tensor): Testing features as tensor.
        y_test_tensor (Tensor): Testing targets as tensor.

    Returns:
        model (nn.Module): Trained neural network model.
    """
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 128  # Number of neurons in the hidden layer
    output_dim = 1
    model = RegressionModel(input_dim, hidden_dim, output_dim)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Calculate training and validation loss
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train_tensor), y_train_tensor).item()
            val_loss = criterion(model(X_test_tensor), y_test_tensor).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Neural Network Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


def own_model_preds(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    """
    Evaluates the trained neural network model and prints RMSE for training and testing sets.

    Args:
        model (nn.Module): Trained neural network model.
        X_train_tensor (Tensor): Training features as tensor.
        y_train_tensor (Tensor): Training targets as tensor.
        X_test_tensor (Tensor): Testing features as tensor.
        y_test_tensor (Tensor): Testing targets as tensor.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_tensor).cpu().numpy()
        y_pred_test = model(X_test_tensor).cpu().numpy()

    # Since no log transformation was applied, use predictions directly
    y_pred_train_actual = y_pred_train.flatten()
    y_pred_test_actual = y_pred_test.flatten()

    # Convert true values to numpy arrays
    y_train_actual = y_train_tensor.cpu().numpy().flatten()
    y_test_actual = y_test_tensor.cpu().numpy().flatten()

    # Calculate MSE and RMSE for train and test sets
    mse_train = mean_squared_error(y_train_actual, y_pred_train_actual)
    mse_test = mean_squared_error(y_test_actual, y_pred_test_actual)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    print(f"Neural Network Model - Training RMSE: {rmse_train:.2f}")
    print(f"Neural Network Model - Testing RMSE: {rmse_test:.2f}")


def main():
    """
    Main function to execute data loading, preparation, modeling, and evaluation.
    """
    specific_occupation = True  # or False, depending on your needs
    handle_outliers = True  # Set to True to handle outliers
    method = 'remove'  # Choose 'remove' or 'cap'

    # Load and prepare data
    df = load_data()
    if df is None:
        print("Data loading failed.")
        return
    print("Data Loaded")

    df = prepare_data(df, specific_occupation=specific_occupation, handle_outliers=handle_outliers, method=method)
    print("Data Prepared")

    # Optional: Preview the prepared data
    print(df[['RAC2P', 'RAC2P_LABEL', 'SCHL', 'SCHL_LABEL', 'SEX', 'SEX_LABEL', 'HISP', 'HISP_LABEL', 'ESR',
              'ESR_LABEL', 'ANC1P', 'ANC1P_LABEL', 'ANC2P', 'ANC2P_LABEL']].head())

    # Feature selection
    X, y = select_features(df, specific_occupation=specific_occupation)
    print("Features Selected")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data Split into Training and Testing Sets")

    # Gradient Boosting Regressor
    GradBoost(X_train, y_train, X_test, y_test)

    # Prepare data for the neural network model
    X_train_scaled, train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_for_model(
        X_train, X_test, y_train, y_test)
    print("Data Prepared for Neural Network Model")

    # Train your own neural network model
    model = own_model(train_loader, X_train_scaled, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

    # Evaluate your own model
    own_model_preds(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)


if __name__ == "__main__":
    main()

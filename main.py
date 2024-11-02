from data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from models import *


df = load_data()

df = prepare_data(df)

X, y = select_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def GradBoost():
    # Initialize the model
    model = GradientBoostingRegressor()


    # Train the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = root_mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def own_model(train_loader, X_train_scaled, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 64  # You can adjust this
    output_dim = 1
    model = RegressionModel(input_dim, hidden_dim, output_dim)

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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def own_model_preds(model, X_train_tensor, X_test_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_tensor).numpy()
        y_pred_test = model(X_test_tensor).numpy()

    # Calculate MSE for train and test sets
    mse_train = np.mean((y_train.values - y_pred_train.flatten()) ** 2)
    mse_test = np.mean((y_test.values - y_pred_test.flatten()) ** 2)

    print(f"Training MSE: {mse_train}")
    print(f"Testing MSE: {mse_test}")


if __name__ == "__main__":
    X_train_scaled, train_loader, X_train_tensor ,y_train_tensor, X_test_tensor , y_test_tensor  = prepare_for_model(X_train, X_test, y_train, y_test)
    own_model(train_loader,X_train_scaled,  X_train_tensor ,y_train_tensor,
               X_test_tensor , y_test_tensor)
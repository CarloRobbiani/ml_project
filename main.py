from data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

df = load_data()

df = prepare_data(df)

X, y = select_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize the model
model = GradientBoostingRegressor()


# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate MSE
mse = root_mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
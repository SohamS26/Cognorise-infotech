import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data1.csv')

# Selecting relevant features for the model
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']
target = 'price'

# Split the dataset into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(random_state=42),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# Train and evaluate the models
results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store the results
    results[model_name] = {
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred)
    }
    
    # Print the evaluation metrics
    print(f"--- {model_name} ---")
    print(f"Mean Absolute Error: {results[model_name]['mae']:.2f}")
    print(f"Mean Squared Error: {results[model_name]['mse']:.2f}")
    print(f"Root Mean Squared Error: {results[model_name]['rmse']:.2f}")
    print(f"R^2 Score: {results[model_name]['r2']:.2f}")
    print()

# Plotting actual vs predicted prices for each model
plt.figure(figsize=(16, 12))

for i, (model_name, result) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, result["y_pred"], color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title(f'Actual vs Predicted Prices ({model_name})')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.grid(True)

plt.tight_layout()
plt.show()

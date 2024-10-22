import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Import the necessary libraries and download the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Display a summary of the dataset
print("Summary of the Boston Housing dataset:")
print(data.describe())
print("\nDataset Head:")
print(data.head())

# Step 2: Prepare X, Y using Train-Test Split
X = data.drop(columns=['medv'])  # 'medv' is the target column
y = data['medv']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build Model using Lasso for Feature Selection

# Initialize Lasso regression model with the maximum number of features
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Get the coefficients and feature names
coefficients = lasso_model.coef_
feature_names = X.columns

# Create a DataFrame to store feature names and their coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Filter the features with non-zero coefficients and sort by absolute value
selected_features = coef_df[coef_df['Coefficient'] != 0]
selected_features = selected_features.reindex(selected_features['Coefficient'].abs().sort_values(ascending=False).index)

# Step 4: Evaluate Model with Different Numbers of Variables
mse_results = []
variables_used = []

# Loop through different numbers of variables to use (from 1 to the number of features available)
for n in range(1, len(feature_names) + 1):
    # Select top n features
    top_n_features = selected_features.head(n)['Feature'].values
    
    # Reduce X_train and X_test to only the selected features
    X_train_n = X_train[top_n_features]
    X_test_n = X_test[top_n_features]
    
    # Initialize and train the Lasso regression model with selected features
    lasso_model_n = Lasso(alpha=1.0)
    lasso_model_n.fit(X_train_n, y_train)
    
    # Predict on the test set
    y_test_pred_n = lasso_model_n.predict(X_test_n)
    
    # Calculate MSE for the test set
    test_mse = mean_squared_error(y_test, y_test_pred_n)
    
    # Store results
    mse_results.append(test_mse)
    variables_used.append(top_n_features)

# Print MSE results and corresponding variable names used in the models
for i in range(len(mse_results)):
    print(f"Number of Variables: {i + 1}, MSE: {mse_results[i]:.2f}, Variables: {variables_used[i]}")

# Plot MSE against the number of variables
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mse_results) + 1), mse_results, marker='o', color='b', linestyle='-')
plt.xticks(range(1, len(mse_results) + 1))
plt.xlabel('Number of Variables')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs Number of Variables in Lasso Regression')
plt.grid()
plt.show()

# Step 5: Predict Y_test Value using the Lasso model with the best number of selected features
best_n = np.argmin(mse_results) + 1  # Get the best number of features
best_features = variables_used[best_n - 1]  # Best variable names

X_train_best = X_train[best_features]
X_test_best = X_test[best_features]

# Initialize and train the Lasso regression model with the best features
lasso_model_best = Lasso(alpha=1.0)
lasso_model_best.fit(X_train_best, y_train)

# Predict on the test set using the best model
y_pred_best = lasso_model_best.predict(X_test_best)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line for reference
plt.xlabel('Actual MEDV Values')
plt.ylabel('Predicted MEDV Values')
plt.title('Actual vs Predicted MEDV Values')
plt.grid()
plt.show()

# Print Predicted vs Actual values for the first 5 test samples
print("\nPredicted vs Actual values for the first 5 test samples:")
for i in range(5):
    print(f"Predicted: {y_pred_best[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")

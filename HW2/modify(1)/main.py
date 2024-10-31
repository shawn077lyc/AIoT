# Step 1: Import Libraries
import os
import re
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

# Step 2: Data Preprocessing
# Handling missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Dropping unnecessary columns
train_data = train_data.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test_data = test_data.drop(['Name', 'Cabin', 'Ticket'], axis=1)

# Encoding categorical variables
label_encoder = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = label_encoder.fit_transform(train_data[col])

for col in test_data.columns:
    if test_data[col].dtype == 'object':
        test_data[col] = label_encoder.fit_transform(test_data[col])

# Splitting features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Step 3: Model Training
# Initialize the DecisionTreeClassifier with hyperparameter tuning
dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best model from GridSearch
best_dt = grid_search.best_estimator_

# Train the best model on the full training data
best_dt.fit(X, y)

# Predictions on the test dataset
predictions = best_dt.predict(test_data)

# Step 4: Prepare the Submission File
submission = pd.DataFrame({'PassengerId': gender_submission['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

# Output the first few rows of the submission file
submission.head()

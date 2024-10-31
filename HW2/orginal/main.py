# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score  # Import cross_val_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Step 3: Handle Missing Values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Step 4: Encode Categorical Variables
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 5: Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Step 6: Drop Unnecessary Columns
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
passenger_ids = test_data['PassengerId']  # Save PassengerId for submission
test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Step 7: Split the Data into Features and Target
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

# Step 8: SelectKBest
def select_k_best_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features

# Step 9: Recursive Feature Elimination (RFE)
def rfe_feature_selection(X, y, n_features_to_select=5):
    model = LogisticRegression(max_iter=200)
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

# Step 10: Use Optuna for Hyperparameter Tuning and Feature Selection
def objective(trial):
    # Feature Selection
    k = trial.suggest_int('k', 3, X_train.shape[1])
    rfe_n_features = trial.suggest_int('rfe_n_features', 3, X_train.shape[1])
    
    # Combine features selected by SelectKBest and RFE
    select_k_features = select_k_best_features(X_train, y_train, k=k)
    rfe_features = rfe_feature_selection(X_train, y_train, n_features_to_select=rfe_n_features)
    
    selected_features = list(set(select_k_features) | set(rfe_features))
    X_selected = X_train[selected_features]
    
    # Model Training and Cross-Validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='accuracy')
    
    return scores.mean()

# Step 11: Run Optuna Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Step 12: Get Best Features
best_k = study.best_params['k']
best_rfe_n_features = study.best_params['rfe_n_features']

# Combine the best features selected by SelectKBest and RFE
final_select_k_features = select_k_best_features(X_train, y_train, k=best_k)
final_rfe_features = rfe_feature_selection(X_train, y_train, n_features_to_select=best_rfe_n_features)
final_selected_features = list(set(final_select_k_features) | set(final_rfe_features))

print("Best Features Selected:", final_selected_features)

# Step 13: Train the Final Model with Best Features
X_train_final = X_train[final_selected_features]
X_test_final = X_test[final_selected_features]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

# Step 14: Prepare Submission File
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

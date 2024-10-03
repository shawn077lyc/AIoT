import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Streamlit app title
st.title('Interactive Linear Regression Example')

# Step 1: User inputs for slope (a), constant (c), and number of points (n) in the sidebar
st.sidebar.header('User Input Parameters')
a = st.sidebar.slider('Select slope (a)', -10.0, 10.0, 0.0)
c = st.sidebar.slider('Select constant (c)', 0.0, 100.0, 50.0)
n = st.sidebar.slider('Select number of points (n)', 100, 500, 250)

# Step 2: Create dataset based on user input
X = np.random.rand(n, 1) * 10  # Feature: Random values between 0 and 10
y = a * X + c + np.random.rand(n, 1) * 10  # Target variable with noise

# Step 3: Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict the values
predictions = model.predict(X)

# Step 5: Create a DataFrame
df = pd.DataFrame(np.hstack((X, y)), columns=['Hours_Studied', 'Test_Scores'])

# Step 6: Create columns for plot and dataset
col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

# Plotting in the first column
with col1:
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')  # Scatter plot of original data
    plt.plot(X, predictions, color='red', label='Regression Line', linewidth=2)  # Regression line
    plt.title('Linear Regression Fit')
    plt.xlabel('Hours Studied')
    plt.ylabel('Test Scores')
    plt.legend()
    plt.grid()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Create a new row for dataset and coefficients
col3, col4 = st.columns([3, 1])  # New columns for dataset and coefficients

with col3:
    st.subheader('Generated Dataset')
    st.write(df)

with col4:
    st.subheader('Model Coefficients')
    st.write(f'Coefficient (slope): {model.coef_[0][0]}')
    st.write(f'Intercept: {model.intercept_[0]}')

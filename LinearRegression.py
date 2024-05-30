from sklearn.linear_model import LinearRegression
import numpy as np

# Example data for Simple Linear Regression
X_simple = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (independent variable)
y_simple = np.array([2, 3, 4, 5, 6])  # Target (dependent variable)

# Example data for Multiple Linear Regression
X_multiple = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Features (independent variables)
y_multiple = np.array([3, 4, 5, 6, 7])  # Target (dependent variable)

# Simple Linear Regression
simple_linear_regression = LinearRegression()
simple_linear_regression.fit(X_simple, y_simple)

# Multiple Linear Regression
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_multiple, y_multiple)

# Predictions
X_simple_test = np.array([6]).reshape(-1, 1)  # New data for Simple Linear Regression
X_multiple_test = np.array([[6, 7]])  # New data for Multiple Linear Regression

prediction_simple = simple_linear_regression.predict(X_simple_test)
prediction_multiple = multiple_linear_regression.predict(X_multiple_test)

print("Simple Linear Regression Prediction:", prediction_simple)
print("Multiple Linear Regression Prediction:", prediction_multiple)

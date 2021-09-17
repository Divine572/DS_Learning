import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('./Datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Position level vs Salary (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary level')
plt.show()


# Visualizing the polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Position level vs Salary (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary level')
plt.show()


# Predicting a new result with Linear Regression
y_lin_pred = lin_reg.predict([[6.5]]) # predicts salary where the position level is 6.5 entered as a 2D array
print(y_lin_pred)




# Predicting a new result with Polynomial Regression

y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(y_poly_pred)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv('./Datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)



# Training the Decision tree regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)  # n_estimators determines the number of trees we want the random forest algorithm to start with ( advisable to start with 10 trees)
regressor.fit(X, y)



# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)


# Visualizing the Decision tree regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position level vs Salary (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary level')
plt.show()



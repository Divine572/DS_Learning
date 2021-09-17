import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv('./Datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)



# Training the Decision tree regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)


# Visualizing the Decision tree regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position level vs Salary (Decision tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary level')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

data = pd.read_csv("Advertising1.csv")
print(data.head())

data.drop(['Unnamed: 0'], axis=1, inplace=True)
print(data.head())

def scatter_plot(feature, target):
    plt.figure(figsize=(16, 8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
        )
    plt.xlabel("Money spent on {} ads ".format(feature))
    plt.ylabel("Sale ")
    plt.show()
    
scatter_plot('TV', 'sales')
scatter_plot('radio', 'sales')
scatter_plot('newspaper','sales')

Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)
mean_MSEs = np.mean(MSEs)
print(mean_MSEs)


ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10,20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(Xs, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


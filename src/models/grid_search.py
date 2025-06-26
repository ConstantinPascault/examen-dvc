
import sklearn
import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

model = Ridge()

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

#--Train the model
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

#--Save the trained model to a file
model_filename = './models/best_params.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_params, f)
print("Grid search trained and saved successfully.")



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
#y_train = np.ravel(y_train)
#y_test = np.ravel(y_test)

bestparams_filename = './models/best_params.pkl'
with open(bestparams_filename, 'rb') as f:
    best_params = pickle.load(f)

model = Ridge(**best_params)

#--Train the model
model.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved successfully.")


import pandas as pd 
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
#y_train = np.ravel(y_train)
#y_test = np.ravel(y_test)

def main(repo_path):
    with open(repo_path / "models/trained_model.pkl", 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(X_test)

    df_final = X_test.copy()
    df_final['pred'] = predictions
    output_filepath = repo_path / "data/predictions/predictions.csv"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_filepath, index=False)

    mse = mean_squared_error(y_test, predictions)
    r2 = model.score(X_test, y_test)
    metrics = {"mse": mse, 'r2': r2}
    scores_path = repo_path / "metrics/scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
import os
import joblib
import json
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

from Linear_regression_01.entity.config_entity import ModelEvaluationConfig
from Linear_regression_01.logging import logger

class ModelEvaluation:

    def __init__(self,config:ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)

        return mae, mse, rmse, r2

    def save_metrics(self, metrics):
        os.makedirs(self.config.root_dir, exist_ok=True)

        with open(self.config.metric_file_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def initiate_model_evaluation(self):

        # Load trained model
        model = joblib.load(self.config.trained_model_file_path)

        # Load transformed test data
        test_arr = np.load(self.config.transformed_test_path)

        # Split X and y
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        # Prediction
        prediction = model.predict(X_test)

        # Calculate metrics
        mae, mse, rmse, r2 = self.eval_metrics(y_test, prediction)

        metrics = {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2 Score": float(r2)
        }

        # Save metrics
        self.save_metrics(metrics)

        # Check threshold
        if r2 >= self.config.threshold:
            print("Model Accepted")
            return True

        else:
            print("Model Rejected")
            return False    
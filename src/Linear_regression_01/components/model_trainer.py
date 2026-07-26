import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from Linear_regression_01.entity.config_entity import ModelTrainerConfig
from Linear_regression_01.logging import logger


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):

        logger.info("Loading transformed training and testing data")

        train_arr = np.load(self.config.transformed_train_path)  # data from transformation in np array format
        test_arr = np.load(self.config.transformed_test_path)

        # Split Features and Target

        X_train = train_arr[:, :-1]   #Take all rows Take all columns Except last
        y_train = train_arr[:, -1]    #Take all rows and only last column i.e, price

        X_test = test_arr[:, :-1]    # take all rows and all columns except last column i.e, price
        y_test = test_arr[:, -1]     # takes all rows and only last colum target varaliable price

        logger.info("Training Linear Regression Model")

        # Train Model

        model = LinearRegression()   #selecting model 

        model.fit(X_train, y_train)  # fitting best model

        logger.info("Model Training Completed")

        # Prediction

        train_prediction = model.predict(X_train)   
        test_prediction = model.predict(X_test)

        # Evaluation

        train_score = r2_score(y_train, train_prediction)  # checcking accuracy 
        test_score = r2_score(y_test, test_prediction)

        logger.info(f"Training R2 Score : {train_score}")
        logger.info(f"Testing R2 Score : {test_score}")

        # Save Model

        os.makedirs(
            os.path.dirname(self.config.trained_model_file_path),
            exist_ok=True
        )

        joblib.dump(
            model,                                 # saving model in model.pkl
            self.config.trained_model_file_path
        )

        logger.info("Model Saved Successfully")

        return model
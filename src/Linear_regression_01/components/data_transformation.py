import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from Linear_regression_01.entity.config_entity import DataTransformationConfig
from Linear_regression_01.logging import logger

def binary_encoding(X):

    X = X.copy()

    mapping = {
        "yes": 1,
        "no": 0
    }

    return X.replace(mapping)

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def initiate_data_transformation(self):

        logger.info("***** Data Transformation Started *****")

        # Check Validation Status


        with open(self.config.STATUS_FILE,"r") as f:

            status = f.read().split(" ")[-1]

        if status != "True":
            raise Exception("Data Validation Failed")

        logger.info("Validation Successful")


        # -----------------------------
        # Read Dataset
        # -----------------------------

        df = pd.read_csv(self.config.unzip_data_dir)

        logger.info(f"Dataset Shape : {df.shape}")


        # -----------------------------
        # Split Features and Target
        # -----------------------------

        X = df.drop(columns=["price"],axis=1)

        y = df["price"]


        # -----------------------------
        # Train Test Split
        # -----------------------------

        X_train,X_test,y_train,y_test = train_test_split(

            X,
            y,
            test_size=0.20,
            random_state=42

        )

        logger.info("Train Test Split Completed")


        # -----------------------------
        # Columns
        # -----------------------------

        binary_columns = [

            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea"

        ]

        categorical_column = [

            "furnishingstatus"

        ]


        # -----------------------------
        # Pipelines
        # -----------------------------

        binary_pipeline = Pipeline(

            steps=[

                ("binary",FunctionTransformer(binary_encoding))

            ]

        )


        categorical_pipeline = Pipeline(

            steps=[

                ("onehot",OneHotEncoder(handle_unknown="ignore"))

            ]

        )


        # -----------------------------
        # Column Transformer
        # -----------------------------

        preprocessor = ColumnTransformer(

            transformers=[

                ("binary",binary_pipeline,binary_columns),

                ("categorical",categorical_pipeline,categorical_column)

            ],

            remainder="passthrough"

        )


        logger.info("Applying Preprocessing")


        X_train = preprocessor.fit_transform(X_train)

        X_test = preprocessor.transform(X_test)


        logger.info("Preprocessing Completed")
        # -----------------------------
        # Save Preprocessor
        # -----------------------------

        joblib.dump(preprocessor, self.config.preprocessor_path)

        logger.info("Preprocessor Saved")


        # -----------------------------
        # Combine Features and Target
        # -----------------------------

        train_arr = np.c_[X_train, np.array(y_train)]

        test_arr = np.c_[X_test, np.array(y_test)]


        # -----------------------------
        # Save Train and Test Arrays
        # -----------------------------

        np.save(self.config.transformed_train_path, train_arr)

        np.save(self.config.transformed_test_path, test_arr)

        logger.info("Transformed Data Saved")


        return (
        self.config.transformed_train_path,
        self.config.transformed_test_path,
        self.config.preprocessor_path

    )
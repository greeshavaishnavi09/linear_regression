import os
import pandas as pd
from Linear_regression_01.logging import logger
from Linear_regression_01.entity.config_entity import DataValidationConfig

class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):

        validation_status = True

        # Read Dataset
        if not os.path.exists(self.config.unzip_data_dir):
            logger.info("Dataset Not Found")
            return False

        df = pd.read_csv(self.config.unzip_data_dir)

        # Dataset Shape
        rows, columns = df.shape
        logger.info(f"Rows : {rows}")
        logger.info(f"Columns : {columns}")

        if rows == 0:
            logger.info("Dataset is Empty")
            validation_status = False

        # Expected Columns and Data Types
        expected_dtype = {
            "price": "int64",
            "area": "int64",
            "bedrooms": "int64",
            "bathrooms": "int64",
            "stories": "int64",
            "mainroad": "object",
            "guestroom": "object",
            "basement": "object",
            "hotwaterheating": "object",
            "airconditioning": "object",
            "parking": "int64",
            "prefarea": "object",
            "furnishingstatus": "object"
        }

        # Schema Validation
        expected_columns = list(expected_dtype.keys())
        actual_columns = list(df.columns)

        if expected_columns == actual_columns:
            logger.info("Schema Validation Passed")
        else:
            logger.info("Schema Validation Failed")
            validation_status = False

        # Missing Values
        missing_values = df.isnull().sum()
        logger.info(missing_values)

        if missing_values.sum() > 0:
            logger.info("Missing Values Found")
            validation_status = False

        # Duplicate Rows
        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate Rows : {duplicates}")

        if duplicates > 0:
            validation_status = False

        # Data Type Validation
        logger.info(df.dtypes)

        actual_dtype = df.dtypes.astype(str).to_dict()

        for column, dtype in expected_dtype.items():
            if actual_dtype[column] != dtype:
                logger.info(f"{column} datatype mismatch")
                validation_status = False

        # Target Column Check
        if "price" not in df.columns:
            logger.info("Target Column Missing")
            validation_status = False

        # Yes / No Column Validation
        yes_no_columns = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea"
        ]

        for col in yes_no_columns:
            if not df[col].isin(["yes", "no"]).all():
                logger.info(f"{col} contains invalid values")
                validation_status = False

        # Furnishing Status Validation
        valid_furnishing = [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]

        if not df["furnishingstatus"].isin(valid_furnishing).all():
            logger.info("Invalid furnishingstatus values")
            validation_status = False

        # Negative Value Check
        numeric_columns = [
            "price",
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking"
        ]

        for col in numeric_columns:
            if (df[col] < 0).any():
                logger.info(f"{col} contains negative values")
                validation_status = False

        # Write Validation Status
        with open(self.config.STATUS_FILE, "w") as f:
            f.write(f"Validation Status : {validation_status}")

        logger.info(f"Validation Status : {validation_status}")

        return validation_status
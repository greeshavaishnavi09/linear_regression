from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.data_transformation import DataTransformation
from Linear_regression_01.logging import logger


STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        data_transformation_config = (
            config.get_data_transformation_config()
        )

        data_transformation = DataTransformation(
            config=data_transformation_config
        )

        data_transformation.initiate_data_transformation()
from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.data_validatation import DataValidation
from Linear_regression_01.logging import logger

STAGE_NAME="DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        validation_config = config.get_data_validation_config()

        validation = DataValidation(validation_config)

        validation.validate_dataset()

obj = DataValidationTrainingPipeline()

obj.main()        
from Linear_regression_01.components.data_ingestion import DataIngestion
from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.data_validatation import DataValidation
from Linear_regression_01.pipeline.stage_01_dataingestion import DataIngestionTrainingPipeline
from Linear_regression_01.pipeline.stage_02_datavalidation import DataValidationTrainingPipeline
from Linear_regression_01.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def main(self):

        config = ConfigurationManager()

        validation_config = config.get_data_validation_config()

        validation = DataValidation(validation_config)

        validation.validate_dataset()

obj = DataValidationTrainingPipeline()

obj.main() 
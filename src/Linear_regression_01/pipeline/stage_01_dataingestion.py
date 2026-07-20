from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.data_ingestion import DataIngestion
from Linear_regression_01.logging import logger


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>>> Data Ingestion Stage Started <<<<<<")

            config = ConfigurationManager()

            data_ingestion_config = config.get_data_ingestion_config()

            data_ingestion = DataIngestion(config=data_ingestion_config)

            data_ingestion.download_files()

            data_ingestion.extract_zip_file()

            logger.info(">>>>>> Data Ingestion Stage Completed <<<<<<")

        except Exception as e:
            logger.exception(e)
            raise e
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

from Linear_regression_01.logging import logger



class DataIngestion:

    def __init__(self, config):
        self.config = config

    def download_files(self):

        if not os.path.exists(self.config.local_data_file):

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                dataset=self.config.dataset_name,
                path=self.config.root_dir,
                unzip=False
            )

            logger.info("Dataset downloaded successfully")

        else:
            logger.info("Dataset already exists")

    def extract_zip_file(self):

        os.makedirs(self.config.unzip_dir, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        logger.info("Dataset extracted successfully")
from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.model_trainer import ModelTrainer
from Linear_regression_01.logging import logger


STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_config = config.config.model_trainer

        model_trainer_config = (                   # configuration manager
            config.get_model_trainer_config()
        )

        model_trainer = ModelTrainer(              # components
            config=model_trainer_config
        )

        model_trainer.train()
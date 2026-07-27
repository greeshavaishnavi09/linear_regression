from Linear_regression_01.config.configuration import ConfigurationManager
from Linear_regression_01.components.model_evaluation import ModelEvaluation
from Linear_regression_01.logging import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __int__(self):
        pass

    def main(Self):
        config = ConfigurationManager()

        model_evaluation_config = (
            config.get_model_evaluation_config()
        )

        model_evaluation = ModelEvaluation(
            config = model_evaluation_config
        )

        model_evaluation.initiate_model_evaluation()
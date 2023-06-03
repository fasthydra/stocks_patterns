import logging
import logging.config
import os

import click
import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from src.logger.log_settings import LOGGING_CONFIG
from src.models.clustering import KShapeClusterer, get_clustering_model

load_dotenv()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test_dvc_new")

assert "MLFLOW_S3_ENDPOINT_URL" in os.environ


class KShapeClustererWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model

    def predict(self, context, inputs):
        """This is an abstract function.
        Args:
          context ([type]): MLflow context where the model artifact is stored
          inputs ([type]): the input data to fit into the model
        Returns:
            [type]: the loaded model artifact.
        """
        result = self.model.predict(inputs)
        return result

    def load_context(self, context):
        """This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context: MLflow context where the model artifact is stored.
        """
        model_path = context.artifacts["kshape_model_path"]
        self.model = KShapeClusterer.load(model_path)


@click.command()
@click.option(
    "--in_file",
    required=True,
    help="Input file full path to read the trading data",
)
@click.option(
    "--model_prms_yaml",
    type=str,
    help="Путь до yaml-файла с параметрами модели",
)
def main(in_file: str, model_prms_yaml: str, model_name: str = "KShape"):
    """Подбирает параметры для датасета и модели кластеризации.

    Args:
         in_file (str): Путь до файла с исходными данными временных рядов.

         model_prms_yaml (str): Путь до yaml-файла с параметрами модели.
            Данный файл должен содержать следующие параметры:
                max_iter,
                n_init,
                n_clusters,
    """

    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug("Начало кластеризации")

    try:
        input_df = np.load(in_file)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        with open(model_prms_yaml) as f:
            model_prms = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Файл {model_prms_yaml} не найден.")
    except yaml.YAMLError as e:
        print(f"Ошибка чтения YAML-файла: {model_prms_yaml}", e)

    model = get_clustering_model(model_name, model_prms)
    model.fit(input_df)
    y_pred = model.predict(input_df)
    score = model.get_metric_std(input_df, y_pred)

    mlflow.log_metrics({"std": score})

    model_path = f"./models/{model_name}"
    model.save(model_path)

    artifacts = {"kshape_model_path": model_path}

    signature = infer_signature(input_df, y_pred)

    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=KShapeClustererWrapper(),
        code_path=["./src/models/clustering.py"],
        artifacts=artifacts,
        registered_model_name=model_name,
        signature=signature,
    )


if __name__ == "__main__":
    main()

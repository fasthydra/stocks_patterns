import logging
import logging.config
import os

import click
import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv

from src.logger.log_settings import LOGGING_CONFIG
from src.models.optimize import optimize

load_dotenv()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test_dvc_new")

assert "MLFLOW_S3_ENDPOINT_URL" in os.environ


@click.command()
@click.option(
    "--in_file",
    required=True,
    help="Input file full path to read the trading data",
)
@click.option("--model_name", required=True, type=str, help="Имя модели")
@click.option("--max_iter", required=True, type=int, help="???")
@click.option("--n_init", required=True, type=int, help="???")
@click.option("--n_clusters", required=True, type=int, help="???")
@click.option(
    "--opt_min",
    required=True,
    type=int,
    help="Минимальное число кластеров для подбора",
)
@click.option(
    "--opt_max",
    required=True,
    type=int,
    help="Максимальное число кластеров для подбора",
)
@click.option(
    "--opt_step",
    required=True,
    type=int,
    help="Шаг для подбора количества кластеров",
)
@click.option(
    "--best_clusters",
    required=True,
    type=int,
    help="Число хороших кластеров, которые ищет модель.",
)
@click.option("--n_trials", type=int, help="Количество проходов оптимизации")
@click.option("--par_file", type=str, help="Путь для сохранения параметров")
def main(
    in_file: str,
    model_name: str,
    max_iter: int,
    n_init: int,
    n_clusters: int,
    opt_min: int,
    opt_max: int,
    opt_step: int,
    best_clusters: int,
    n_trials: int,
    par_file: str,
):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug("Начало подбора параметров.")

    try:
        input_df = np.load(in_file)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        best_params = optimize(
            data=input_df,
            model_name=model_name,
            model_prmt={
                "max_iter": max_iter,
                "n_init": n_init,
                "n_clusters": n_clusters,
            },
            opt_clusters_min_max_step=(opt_min, opt_max, opt_step),
            best_cl=best_clusters,
            n_trials=n_trials,
        )
        logger.info("Параметры модели подобраны.")
        logger.info(best_params)
    except Exception as ex:
        logger.exception(f"Ошибка при подборе параметров: {ex}")

    try:
        with open(par_file, "w") as f:
            yaml.dump(
                best_params, f, sort_keys=False, default_flow_style=False
            )

    except Exception as ex:
        logger.exception(f"Ошибка при сохранении параметров: {ex}")

    with mlflow.start_run():
        mlflow.log_param("my", best_params)


if __name__ == "__main__":
    main()

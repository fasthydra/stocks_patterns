import logging
import logging.config
import os

import click
import mlflow
import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv

from src.logger.log_settings import LOGGING_CONFIG
from src.models.optimize import optimize

load_dotenv(find_dotenv(usecwd=True))
load_dotenv()

assert "MLFLOW_S3_ENDPOINT_URL" in os.environ
assert "MLFLOW_EXPERIMENT_NAME" in os.environ


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
@click.option(
    "--trials_file",
    type=str,
    help="Путь для сохранения результатов экспериментов",
)
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
    trials_file: str,
):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug("Начало подбора параметров.")

    try:
        input_df = np.load(in_file)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    try:
        best_params, trials_df = optimize(
            data=input_df,
            model_name=model_name,
            model_prmt={
                "max_iter": max_iter,
                "n_init": n_init,
                "n_clusters": n_clusters,
            },
            opt_clusters_min_max_step=(opt_min, opt_max, opt_step),
            best_clusters=best_clusters,
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
        mlflow.log_artifact(par_file)
    except Exception as ex:
        logger.exception(f"Ошибка при сохранении параметров: {ex}")

    try:
        trials_df.to_csv(sep=",", path_or_buf=trials_file)
        mlflow.log_artifact(trials_file)
    except Exception as ex:
        logger.exception(f"Ошибка при сохранении параметров: {ex}")

    # with mlflow.start_run():
    #     mlflow.log_param("my", best_params)


if __name__ == "__main__":
    main()

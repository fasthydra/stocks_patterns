import logging
import logging.config
import pprint

import click
import dvc.api
import numpy as np
import pandas as pd

from src.data.preprocessing import data_division
from src.logger.log_settings import LOGGING_CONFIG


@click.command()
@click.option(
    "--in_file", help="Input file full path to read the trading data"
)
def main(in_file: str):
    params = dvc.api.params_show()["slice_ts"]

    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("file_logger")

    logger.debug(
        "Параметры запуска: \n" + pprint.pformat(params, compact=True)
    )

    try:
        input_df = pd.read_csv(in_file)
    except Exception as ex:
        logger.exception(f"Ошибка чтения файла: {ex}")

    array_cluster, array_predict, scaler = data_division(
        data=input_df,
        column_value=params["column_value"],
        period=params["period"],
        step=params["step"],
        rolling_period=params["rolling_period"],
        predict_period=params["predict_period"],
        train_sample=params["train_sample"],
    )

    try:
        file_to_save = params["paths"]["output"]["cluster"]
        np.save(file_to_save, array_cluster)
        logger.info(
            f"Данные (строк: {array_cluster.shape[0]}) успешно загружены /"
            f"в файл {file_to_save}"
        )
    except Exception as ex:
        logger.exception(
            f"Ошибка при сохранении датасета в файл {file_to_save}: /" f" {ex}"
        )

    try:
        file_to_save = params["paths"]["output"]["predict"]
        array_predict.to_csv(sep=",", path_or_buf=file_to_save)
        logger.info(
            f"Данные (строк: {array_predict.shape[0]}) успешно загружены /"
            f"в файл {file_to_save}"
        )
    except Exception as ex:
        logger.exception(
            f"Ошибка при сохранении датасета в файл {file_to_save}: /" f" {ex}"
        )


if __name__ == "__main__":
    main()

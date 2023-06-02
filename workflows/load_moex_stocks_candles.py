import logging
import logging.config
import pprint

import dvc.api
import mlflow
from dotenv import load_dotenv

from src.data.load_from_moex import load_stocks
from src.logger.log_settings import LOGGING_CONFIG

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test_dvc_new")

params = dvc.api.params_show()["load_moex"]

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("file_logger")

logger.debug("Параметры запуска: \n" + pprint.pformat(params, compact=True))

mlflow.log_params(params)

df = load_stocks(
    security=params["security"],
    interval=params["interval"],
    start=params["start"],
    end=params["end"],
)

try:
    file_to_save = params["paths"]["output"]
    df.to_csv(sep=",", path_or_buf=file_to_save)
    logger.info(
        f"Данные (строк: {df.shape[0]}) успешно загружены /"
        f"в файл {file_to_save}"
    )
    mlflow.log_artifact(file_to_save)
except Exception as ex:
    logger.exception(
        f"Ошибка при сохранении датасета в файл {file_to_save}: /" f" {ex}"
    )

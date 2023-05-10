import logging
import logging.config
import pprint

import dvc.api

from src.data.load_from_moex import load_stocks
from src.logger.log_settings import LOGGING_CONFIG

params = dvc.api.params_show()["load_moex"]

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("file_logger")

logger.debug("Параметры запуска: \n" + pprint.pformat(params, compact=True))

df = load_stocks(
    security=params["security"],
    interval=params["interval"],
    start=params["start"],
    end=params["end"],
)

try:
    df.to_csv(sep=",", path_or_buf=params["paths"]["output"])
    logger.info(
        f"Данные (строк: {df.shape[0]}) успешно загружены /"
        f'в файл {params["paths"]["output"]}'
    )
except Exception as ex:
    logger.exception(
        f'Ошибка при сохранении датасета в файл {params["paths"]["output"]}: /'
        f" {ex}"
    )

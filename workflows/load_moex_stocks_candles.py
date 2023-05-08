import logging
import logging.config

import dvc.api

from src.data.load_from_moex import load_stocks
from src.logger.log_settings import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("load_moex_stocks_candles")

params = dvc.api.params_show()["load_moex"]

logger.debug("Параметры запуска: \n" + params)

df = load_stocks(
    security=params["security"],
    interval=params["interval"],
    start=params["start"],
    end=params["end"],
)

try:
    df.to_csv(sep=",", path_or_buf=params["paths"]["output"])
    logger.info(f'Данные успешно загружены: {params["paths"]["output"]}')
except Exception as ex:
    logger.exception(
        f'Ошибка при сохранении датасета в файл {params["paths"]["output"]}: /'
        f" {ex}"
    )

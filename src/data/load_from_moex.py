import logging

import apimoex
import pandas as pd
import requests

logger = logging.getLogger("file_logger")


def load_stocks(
    security: str,
    start: str,
    end: str,
    interval: int,
):
    """
    Возвращает Pandas.DataFrame, содержащий данные торгов
    :param security: Тикер ценной бумаги
        (https://www.moex.com/ru/listing/securities-list.aspx)
        в столбце Торговый код
    :param start: Дата вида ГГГГ-ММ-ДД
    :param end: Дата вида ГГГГ-ММ-ДД
    :param interval:
        Размер свечки - целое число 1 (1 минута), 10 (10 минут),
        60 (1 час), 24 (1 день), 7 (1 неделя), 31 (1 месяц) или
        4 (1 квартал). По умолчанию дневные данные.
    :return Pandas.DataFrame:
        данные торгов в структуре: begin,open,close,high,low,value
    """
    with requests.Session() as session:
        try:
            data = apimoex.get_market_candles(
                session,
                interval=interval,
                security=security,
                start=start,
                end=end,
            )
            logger.debug("Получили данные биржи. Строк: " + str(len(data)))
        except Exception as ex:
            logger.exception(f"Ошибка: {ex}")

    df = pd.DataFrame(data)
    if len(df):
        df["begin"] = pd.to_datetime(df["begin"])
        df.set_index("begin", inplace=True)

    return df

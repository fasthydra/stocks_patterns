from pathlib import Path

import apimoex
import click
import pandas as pd
import requests


@click.command()
@click.option("--name", default="data", help="File name to save the data")
@click.option("--start", default="2021-01-01", help="Start date")
@click.option("--end", default="2022-01-01", help="Closing date")
@click.option("--interval", default=24, help="Timeframe")
@click.option("--security", default="SBER", help="Stock ticket")
def load_data_from_apimoex(
    name: str,
    start: str,
    end: str,
    interval: int,
    security: str,
):
    """
    Сохраняет данные в папку data/raw проекта в формате csv
    :param name: Название файла
    :param start: Дата вида ГГГГ-ММ-ДД
    :param end: Дата вида ГГГГ-ММ-ДД
    :param interval:
        Размер свечки - целое число 1 (1 минута), 10 (10 минут),
        60 (1 час), 24 (1 день), 7 (1 неделя), 31 (1 месяц) или
        4 (1 квартал). По умолчанию дневные данные.
    :param security: Тикер ценной бумаги
        (https://www.moex.com/ru/listing/securities-list.aspx)
        в столбце Торговый код
    :return
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
        except Exception as ex:
            print(f"Ошибка: {ex}")

        try:
            df = pd.DataFrame(data)
            df["begin"] = pd.to_datetime(df["begin"])
            df.set_index("begin", inplace=True)
            file_to_save = (
                Path(__file__).resolve().parents[2] / f"data/raw/{name}.csv"
            )
            df.to_csv(sep=",", path_or_buf=file_to_save)
            print(f"Данные успешно загружены: {file_to_save}")
        except Exception as ex:
            print(f"Ошибка при формировании датасета: {ex}")


if __name__ == "__main__":
    load_data_from_apimoex()

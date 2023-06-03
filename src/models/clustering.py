import logging
import pickle

import numpy as np
from tslearn.clustering import KShape

from src.models.metrics import metric_std

logger = logging.getLogger("file_logger")


def get_clustering_model(model_name: str, model_prmt: dict) -> object:
    """Инициализирует модель с заданными параметрами.

    Args:
        model_name (str): Имя модели кластеризации.
        model_prmt (dict): Словарь с параметрами модели.

    Returns:
        object: Экземпляр класса модели.
    """
    if model_name == "KShape":
        return KShapeClusterer(model_prmt)
    else:
        logger.error(f"Неверное имя модели: {model_name}")


class KShapeClusterer:
    """Создает модель кластеризации KShape по заданным параметрам

    Methods:
        fit_predict(data): обучает модель и прогнозирует кластеры
        исходя из обученных центров

        get_metric_std(data, y_pred, best_cl):
        вычисляет СКО для выбранного количества лучших кластеров.

        save(filename): сохраняет объект класса в файл
        load(filename): загружает объект класса из файла

    """

    def __init__(
        self,
        model_prmt: dict,
        seed: int = 0,
    ):
        """Устанавливает seed и параметры для модели KShape.

        Args:

            model_prmt (dict): Словарь с параметрами модели.
            seed (int, optional): Значение для инициализации случайных чисел.
                Defaults to 0.
        """

        assert isinstance(model_prmt, dict), "model_prmt должен быть словарем"
        assert (
            "max_iter" in model_prmt
        ), "Отсутствует ключ 'max_iter' в model_prmt"
        assert "n_init" in model_prmt, "Отсутствует ключ 'n_init' в model_prmt"
        assert (
            "n_clusters" in model_prmt
        ), "Отсутствует ключ 'n_clusters' в model_prmt"

        self.seed = seed

        self.model = KShape(
            max_iter=model_prmt["max_iter"],
            n_init=model_prmt["n_init"],
            n_clusters=model_prmt["n_clusters"],
        )

    def fit(self, data: np.ndarray):
        self.model.fit(data)

    def predict(self, data: np.ndarray):
        return self.model.predict(data)

    def fit_predict(self, data: np.ndarray):
        self.fit(data)
        return self.predict(data)

    def get_metric_std(
        self, data: np.ndarray, y_pred: np.ndarray, best_cl: int = None
    ) -> float:
        score = metric_std(self.model.cluster_centers_, data, y_pred, best_cl)
        return score

    def save(self, filename):
        """Сохраняет объект класса в файл с помощью pickle.

        Args:
            filename (str): Имя файла для сохранения.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Загружает объект класса из файла, созданный с помощью pickle.

        Args:
            filename (str): Имя файла для загрузки.

        Returns:
            KShapeClusterer: Загруженный объект класса KShapeClusterer.
        """
        with open(filename, "rb") as file:
            return pickle.load(file)

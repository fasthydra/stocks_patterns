import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("file_logger")


def std(
    cluster_centers: np.ndarray,
    data: np.ndarray,
    y_pred: np.ndarray,
) -> list:
    """Рассчитывает среднеквадратичное отклонение от центроидов.

    Args:
        cluster_centers (np.ndarray): Центроиды кластеров.

        data (np.ndarray): Стандартизированный датасет.

        y_pred (np.ndarray): Предсказанные кластеры.

    Returns:
        std_cl (list): Список СКО.
    """

    all_cl = cluster_centers.shape[0]
    std_cl = []
    try:
        for i in range(all_cl):
            centr = cluster_centers[i].ravel()
            all_ts = []
            for j in range(len(data[y_pred == i])):
                ts = data[y_pred == i][j].ravel()
                all_ts += [list(ts)]
            all_avg = np.array(pd.DataFrame(all_ts).mean(axis=0))
            diff = (centr - all_avg) ** 2
            std = (np.sum(diff) / (len(diff) - 1)) ** (1 / 2)
            std_cl.append(std)

        return std_cl

    except Exception as ex:
        logger.error(f"Ошибка при расчете СКО: {ex}")
        logger.error(f"Кластер {i}, Строка {j}")


def metric_std(
    cluster_centers: np.ndarray,
    data: np.ndarray,
    y_pred: np.ndarray,
    best_cl: int = 10,
) -> float:
    """Рассчитывает среднее среднеквадратичное отклонение от центроидов
        по {best_cl} лучшим кластерам (зависит от числа кластеров,
        которое минимально хотим найти).

    Args:
        cluster_centers (np.ndarray): Центроиды кластеров.

        data (np.ndarray): Стандартизированный датасет.

        y_pred (np.ndarray): Предсказанные кластеры.

        best_cl (int, optional): Количество кластеров для расчета метрики.
            Defaults to 10.

    Returns:
        mean_std (float): Среднее СКО на лучших кластерах.
    """
    list_std = std(cluster_centers, data, y_pred)
    mean_std = np.mean(np.sort(list_std)[:best_cl])

    return mean_std


def indices_std(
    cluster_centers: np.ndarray,
    data: np.ndarray,
    y_pred: np.ndarray,
    best_cl: int = 10,
) -> list:
    """Выводит индексы кластеров с лучшими СКО.

    Args:
        cluster_centers (np.ndarray): Центроиды кластеров.

        data (np.ndarray): Стандартизированный датасет.

        y_pred (np.ndarray): Предсказанные кластеры.

        best_cl (int, optional): Количество кластеров для расчета метрики.
            Defaults to 10.

    Returns:
        idx (list): Список индексов кластеров с лучшими СКО.
    """
    list_std = std(cluster_centers, data, y_pred)
    idx = list(np.argsort(list_std)[:best_cl])

    return idx

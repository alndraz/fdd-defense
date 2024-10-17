from fdd_defense.attackers.base import BaseAttacker
import numpy as np


class DeepLIFTAttack(BaseAttacker):
    def __init__(self, model, eps, selected_indices, attributions_time, interpretation_threshold):
        """
        Инициализация атаки на основе интерпретации DeepLIFT.

        Parameters:
        ----------
        model: object
            Модель, на которую производится атака.
        eps: float
            Максимальное смещение (шум), которое можно добавить к данным.
        selected_indices: list
            Список индексов признаков, которые будут атакованы.
        attributions_time: dict
            Словарь с важностями признаков на каждом временном шаге для выбранных признаков.
        interpretation_threshold: float
            Порог для интерпретации важности. Признаки с важностью выше этого порога будут атакованы.
        """
        super().__init__(model, eps)
        self.selected_indices = selected_indices
        self.attributions_time = attributions_time
        self.interpretation_threshold = interpretation_threshold

    def attack(self, ts, label):
        """
        Реализация атаки с добавлением шума на основе интерпретации важности признаков.

        Parameters:
        ----------
        ts: np.ndarray
            Массив данных сенсоров для атаки (размер: batch_size, sequence_length, num_sensors).
        label: np.ndarray
            Массив меток для данных сенсоров.

        Returns:
        ----------
        np.ndarray:
            Массив данных после атаки.
        """
        # Копируем исходные данные, чтобы не изменять их напрямую
        perturbed_ts = np.copy(ts)

        # Проходим по каждому временному шагу
        for t in range(perturbed_ts.shape[1]):  # Итерируем по временным шагам
            # Для каждого признака из выбранных индексов проверяем важность на этом шаге
            for index in self.selected_indices:
                # Проверяем, превышает ли важность порог для данного признака на текущем временном шаге
                if abs(self.attributions_time[index][t]) > self.interpretation_threshold:
                    # Генерируем шум для атаки
                    noise = self.eps * np.random.uniform(-1, 1)
                    # Добавляем шум к текущему значению признака на данном временном шаге
                    perturbed_ts[:, t, index] += noise

        return perturbed_ts

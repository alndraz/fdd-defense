from fdd_defense.attackers.base import BaseAttacker
import numpy as np

class DeepLIFTAttack(BaseAttacker):
    def __init__(self, model, eps, selected_indices):
        """
        Инициализация атакера, добавляющего шум к выбранным признакам на всех временных шагах.

        Parameters:
        ----------
        model: object
            Модель, на которую производится атака.
        eps: float
            Максимальное смещение (шум), которое можно добавить к данным.
        selected_indices: list
            Список индексов признаков, которые будут атакованы.
        """
        super().__init__(model, eps)
        self.selected_indices = selected_indices

    def attack(self, ts, label):
        """
        Реализация атаки с добавлением шума выбранным признакам на всех временных шагах.

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
            # Для каждого выбранного признака добавляем шум на всех временных шагах
            for index in self.selected_indices:
                # Генерируем шум для атаки
                noise = self.eps * np.random.choice([1, -1], size=ts.shape[0])
                # Добавляем шум к текущему значению признака на данном временном шаге
                perturbed_ts[:, t, index] += noise

        return perturbed_ts

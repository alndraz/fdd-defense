from torch import nn
from fdd_defense.models.base import BaseTorchModel


class MLP(BaseTorchModel):
    def __init__(
            self,
            window_size: int,   # Параметр window_size
            step_size: int,
            batch_size=128,
            lr=0.001,
            num_epochs=10,
            is_test=False,
            device='cpu',
            hidden_dim=624,
    ):
        super().__init__(
            window_size, step_size, batch_size, lr, num_epochs, is_test, device,
        )
        self.hidden_dim = hidden_dim
        self.device = device
        self.window_size = window_size  # Не забывайте сохранять этот параметр

        # Инициализация архитектуры модели в конструкторе
        num_sensors = 52  # Количество сенсоров (можно изменить на основе данных)
        num_states = 21  # Количество классов (также может зависеть от задачи)

        # Определение модели
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, self.hidden_dim),  # Использование window_size
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_states),
        ).to(self.device)  # Перенос модели на устройство (GPU или CPU)

    def fit(self, dataset):
        super().fit(dataset)
        super()._train_nn()

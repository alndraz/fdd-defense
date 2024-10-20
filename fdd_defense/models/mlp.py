from torch import nn
from fdd_defense.models.base import BaseTorchModel

class MLP(BaseTorchModel):
    def __init__(
            self, 
            window_size: int, 
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

    def fit(self, dataset):
        super().fit(dataset)
        num_sensors = self.dataset.df.shape[1]
        num_states = len(set(self.dataset.label))
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_states),
        )
        super()._train_nn()
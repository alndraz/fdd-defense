import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm


class MLP(nn.Module):
    def __init__(self, window_size, step_size, device='cpu', hidden_dim=2048, num_epochs=2, batch_size=512, lr=0.001):
        super(MLP, self).__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(window_size * 52, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset):
        train_data = dataset.df[dataset.train_mask].values
        batches = [
            train_data[i:i + self.window_size]
            for i in range(0, len(train_data) - self.window_size, self.step_size)
        ]
        batches = torch.tensor(batches, dtype=torch.float32).to(self.device)
        labels = torch.tensor(dataset.labels[dataset.train_mask][self.window_size:], dtype=torch.long).to(self.device)

        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_start in tqdm(range(0, len(batches), self.batch_size)):
                batch = batches[batch_start:batch_start + self.batch_size]
                label_batch = labels[batch_start:batch_start + self.batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch.view(batch.size(0), -1))
                loss = self.loss_fn(outputs, label_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(batches):.4f}")

    # Метод для сохранения весов модели
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Модель сохранена в {path}")

    # Метод для загрузки весов модели
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Модель загружена из {path}")

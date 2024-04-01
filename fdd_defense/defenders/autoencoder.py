import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm, trange
from abc import ABC

from fdd_defense.defenders.base import BaseDefender


class MLPEncoder(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            bottleneck_size: int=3,
        ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        bottleneck = []
        for i in range(bottleneck_size):
            bottleneck.append(nn.Linear(64, 64))
            bottleneck.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
        ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_sensors * window_size)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, self.window_size, self.num_sensors)


class AutoEncoderDefender(BaseDefender, ABC):
    def __init__(self, model, lr, training_attacker, adv_coeff, num_epochs):
        super().__init__(model)
        self.training_attacker = training_attacker
        self.loss = nn.MSELoss(reduction='mean')
        self.lr = lr
        self.adv_coeff = adv_coeff
        self.num_epochs = num_epochs
        self.encoder = None
        self.decoder = None
    
    def train(self):
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        self.encoder.train()
        self.decoder.train()
        print('Autoencoder training...')
        for e in trange(self.num_epochs, desc='Epochs ...'):
            losses = []
            for batch, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                batch_ = torch.FloatTensor(batch).to(self.model.device)
                rec_batch = self.decoder(self.encoder(batch_))
                rec_loss = self.loss(rec_batch, batch_)
                adv_loss = 0
                if self.training_attacker is not None:
                    adv_batch = self.training_attacker.attack(batch, label)
                    batch_ = torch.FloatTensor(adv_batch).to(self.model.device)
                    rec_batch = self.decoder(self.encoder(batch_))
                    adv_loss = self.loss(rec_batch, batch_)
                loss = rec_loss + self.adv_coeff * adv_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')
        self.encoder.eval()
        self.decoder.eval()

    def predict(self, batch: np.ndarray):
        batch = torch.FloatTensor(batch).to(self.model.device)
        with torch.no_grad():
            def_batch = self.decoder(self.encoder(batch))
        def_batch = def_batch.cpu().numpy()
        return self.model.predict(def_batch)


class MLPAutoEncoderDefender(AutoEncoderDefender):
    def __init__(
            self, 
            model, 
            lr=0.01, 
            training_attacker=None, 
            adv_coef=1., 
            num_epochs=10, 
            bottleneck_size=3
        ):
        super().__init__(model, lr, training_attacker, adv_coef, num_epochs)
        num_sensors = self.model.dataset.df.shape[1]
        window_size = self.model.window_size
        self.encoder = MLPEncoder(num_sensors, window_size, bottleneck_size)
        self.decoder = MLPDecoder(num_sensors, window_size)
        super().train()

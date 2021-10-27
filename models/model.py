import numpy as np
from collections import namedtuple
import torch.nn as nn
from torch.optim import Adam
from torch import save, load
from .metrics import accuracy
import pytorch_lightning as pl

RunResult = namedtuple("RunResult", ['train_history', 'val_history'])
Parameters = namedtuple("Parameters", ['lr', 'epoch', 'optim', 'anneal_coef', 'anneal_epoch', 'dim'])


class PlBugLocModel(pl.LightningModule):
    def __init__(self, base_model, loss=nn.CrossEntropyLoss, word_emb_dim=320, lstm_hidden_dim=60, lr=1e-2):
        super().__init__()
        self.model = base_model(word_emb_dim=word_emb_dim,
                lstm_hidden_dim=lstm_hidden_dim)
        self.loss = loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.transpose(2, 1), y)
        acc = accuracy(y, y_hat, 1)
        self.log("Cross-Entropy Loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss, 'acc':acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat.transpose(2, 1), y)
        acc = accuracy(y, y_hat, 1)
        self.log("Cross-Entropy Loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss, 'acc':acc}

    def validation_epoch_end(self, outputs):
        avg_acc = np.mean([i['acc'] for i in outputs])
        self.log("Val/Acc", avg_acc,  prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        avg_acc = np.mean([i['acc'] for i in outputs])

        self.log("Train/Acc", avg_acc, prog_bar=True, logger=True)

    def train_dataloader(self):
        return self.trainset

    def val_dataloader(self):
        return self.validset

    def save_model(self, path='./models/lstm_code2seq_model'):
        save(self.model, path)

    def load_model(self, path='./models/lstm_code2seq_model'):
        self.model = load(path)

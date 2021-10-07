import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
import math
from tqdm import tqdm
import itertools as it
import json
from . import rank_loss
from . import flat_model
from .LSTMTagger import LSTMTagger
from sklearn.model_selection import train_test_split
from .metrics import accuracy, bug_probability, check_code_embeddings, count_embeddings_before_buggy_method
import pytorch_lightning as pl
from .dataset_wrapper import read_data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

RunResult = namedtuple("RunResult", ['train_history', 'val_history'])
Parameters = namedtuple("Parameters", ['lr', 'epoch', 'optim', 'anneal_coef', 'anneal_epoch', 'dim'])

class PlBugLocModel(pl.LightningModule):
    def __init__(self, base_model, loss=nn.CrossEntropyLoss):
        super().__init__()
        self.model = base_model(word_emb_dim=320,
                lstm_hidden_dim=80)
        self.loss = loss()


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.transpose(2, 1), y)
        acc = accuracy(y, y_hat)
        tqdm_dict = {'val_loss':acc}
        return OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

    def val_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.transpose(2, 1), y)
        acc = accuracy(y, y_hat)
        tqdm_dict = {'val_loss':acc}
        return OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})


class BugLocalizationModel():
    lrs = [1e-2, 1e-3]
    epochs = [20]
    anneal_coefs = [0.25]
    anneal_epochs = [5, 10, 15]
    hidden_dim = [40, 60, 80]
    optims = [optim.Adam]

    def __init__(self, ranked=False, flat=False, embeddings_size=320):
        self.embeddings_size = embeddings_size
        self.best_val_acc = 0.0
        self.run_records = []
        self.model = None
        if ranked:
            self.do_epoch = rank_loss.do_epoch
            self.criterion = nn.MarginRankingLoss
        else:
            self.do_epoch = do_epoch
            self.criterion = nn.CrossEntropyLoss

    def create_list_of_train_hyperparameters(self):
        params = it.product(self.lrs, self.epochs, self.optims, self.anneal_coefs,
                            self.anneal_epochs, self.hidden_dim)
        params = map(lambda param: Parameters(*param), params)
        return params

    def train(self, train_dataloader, test_dataloader, params, model_to_train, top_k=2):
        self.run_records = []

        for param in params:
            print(param)
            loss = self.criterion()
            self.model = model_to_train(
                word_emb_dim=self.embeddings_size,
                lstm_hidden_dim=param.dim)

            optimizer = param.optim(self.model.parameters(), lr=param.lr)
            scheduler = StepLR(optimizer, step_size=param.anneal_epoch, gamma=param.anneal_coef)
            train_acc, val_acc = fit(self, loss, optimizer,
                                     train_data=train_dataloader, epochs_count=param.epoch,
                                     batch_size=128, val_data=test_dataloader,
                                     val_batch_size=128, scheduler=scheduler, top_k=top_k)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            self.run_records.append(RunResult(train_acc, val_acc))
        return self.run_records

    def save_results(self, name):
        results = []
        for i, record in enumerate(self.run_records):
            result = {'train_acc': record.train_history, 'val_acc': record.val_history}
            param = self.params[i]._asdict()
            param = dict(map(lambda item: (str(item[0]), str(item[1])), param.items()))
            result.update(param)
            results.append(result)
        results.sort(key=lambda x: -x['val_acc'])
        f = open('results' + name + '.txt', 'w')
        json.dump(results, f, indent=4)
        f.close()

    def best_params(self):
        best_acc = 0
        best_param = None
        for i, record in enumerate(self.run_records):
            if record.val_history > best_acc:
                best_acc = record.val_history
                best_param = self.params[i]

        return best_param

    def save_model(self, path='./models/lstm_code2seq_model'):
        torch.save(self.model, path)

    def load_model(self, path='./models/lstm_code2seq_model'):
        self.model = torch.load(path)

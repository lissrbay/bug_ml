import pandas as pd
import numpy as np
from collections import namedtuple
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

RunResult = namedtuple("RunResult", ['train_history', 'val_history'])
Parameters = namedtuple("Parameters", ['lr', 'epoch', 'optim', 'anneal_coef', 'anneal_epoch', 'dim'])


def read_data(embeddings_path='X.npy', labels_path='y.npy', reports_path=None):
    X = np.load(embeddings_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True)
    if reports_path is None:
        return X, y
    reports_used = np.load(reports_path)
    return X, y, reports_used


def iterate_batches(data, batch_size):
    X, y = data
    n_samples = len(X)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_indices = indices[start:end]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        yield X_batch, y_batch, batch_indices


def do_epoch(model, criterion, data, batch_size, optimizer=None, name=None, top_k=2):
    epoch_loss = 0
    correct_count = 0
    sum_count = 0
    is_train = not optimizer is None
    name = name or ''

    model.model.train(is_train)

    batches_count = math.ceil(len(data[0]) / batch_size)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, (X_batch, y_batch, indices) in enumerate(iterate_batches(data, batch_size)):
                X_batch, y_batch = FloatTensor(X_batch), LongTensor(y_batch)
                logits = model.model(X_batch)

                loss = criterion(logits.transpose(2, 1), y_batch)
                epoch_loss += loss.item()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                correct_count_delta = accuracy(y_batch, logits, top_k)
                correct_count += correct_count_delta * batch_size
                sum_count += batch_size

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
                    name, epoch_loss / (i + 1), correct_count / sum_count)
                )

    return epoch_loss / batches_count, correct_count / sum_count


def fit(model, criterion, optimizer, train_data, epochs_count=1, batch_size=32,
        val_data=None, val_batch_size=None, scheduler=None, top_k=2):
    if not val_data is None and val_batch_size is None:
        val_batch_size = batch_size
    all_train_acc = []
    all_val_acc = []
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, train_acc = model.do_epoch(model, criterion, train_data, batch_size,
                                               optimizer, name_prefix + 'Train:', top_k)

        if not val_data is None:
            val_loss, val_acc = model.do_epoch(model, criterion, val_data, val_batch_size,
                                               None, name_prefix + '  Val:', top_k)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        if scheduler:
            scheduler.step()
    return np.array(all_train_acc).max(), np.array(all_val_acc).max()


def zero_embeddings_data_accuracy(model):
    X, y_true, _ = read_data('unknown_embeddings.npy', 'unknown_embeddings_labels.npy')
    with torch.no_grad():
        logits = model.model(FloatTensor(X))
        y_pred = bug_probability(logits)

    return accuracy(y_true, y_pred)


class BugLocalizationModel():
    lrs = [1e-2, 1e-3]
    epochs = [20]
    anneal_coefs = [0.25]
    anneal_epochs = [5, 10, 15]
    hidden_dim = [40, 60, 80]
    optims = [optim.Adam]

    def __init__(self, embeddings_path=None, labels_path=None,
                 reports_path=None, ranked=False, flat=False):
        if not (embeddings_path is None):
            if reports_path is None:
                self.X, self.y = read_data(embeddings_path, labels_path)
            else:
                self.X, self.y, self.reports_used = read_data(embeddings_path, labels_path, reports_path)
            self.has_code = check_code_embeddings(self.X, self.y)
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.y)
            self.embeddings_size = self.X.shape[2]
        self.best_val_acc = 0.0
        self.run_records = []
        self.model = None
        if ranked:
            self.do_epoch = rank_loss.do_epoch
            self.criterion = nn.MarginRankingLoss
        else:
            self.do_epoch = do_epoch
            self.criterion = nn.CrossEntropyLoss

        if not (embeddings_path == None) and flat:
            (self.train_x, self.train_y, self.test_x,
             self.test_y, self.report_info) = flat_model.test_train_split_flat(self.X, self.y)
            self.do_epoch = flat_model.do_epoch

    def create_list_of_train_hyperparameters(self):
        params = it.product(self.lrs, self.epochs, self.optims, self.anneal_coefs,
                            self.anneal_epochs, self.hidden_dim)
        params = map(lambda param: Parameters(*param), params)
        return params

    def train(self, params, model_to_train, top_k=2):
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
                                     train_data=(self.train_x, self.train_y), epochs_count=param.epoch,
                                     batch_size=128, val_data=(self.test_x, self.test_y),
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

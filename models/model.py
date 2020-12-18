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
import rank_loss
import flat_model


RunResult = namedtuple("RunResult", ['train_history', 'val_history'])
Parameters = namedtuple("Parameters", ['lr', 'epoch', 'optim', 'anneal_coef', 'anneal_epoch', 'dim'])

def read_data(embeddings_path='X.npy', labels_path='y.npy', reports_path='report_ids.npy'):
    X = np.load(embeddings_path)
    y = np.load(labels_path)
    reports_used = np.load(reports_path)
    return X, y, reports_used


def train_test_split(X, y, train_size = 0.8):
    assert X.shape[0] == y.shape[0]
    train_samples = int(train_size*X.shape[0])
    train_x, train_y = X[:train_samples], y[:train_samples]
    test_x, test_y = X[train_samples:], y[train_samples:]
    return train_x, train_y, test_x, test_y


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
        
        
class ExtendedEmbedding:
    def __init__(self, vocab_size = 2, word_emb_dim = 384):
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        
    def embeddings(self, inputs):
        emb = self.word_emb(torch.LongTensor([[0]]))
        inputs[np.where(~inputs.detach().numpy().all(axis=2))] = emb
        emb = self.word_emb(torch.LongTensor([[1]]))
        inputs[np.where(np.sum(inputs.detach().numpy(), axis=2)==self.word_emb_dim)] = emb
        return inputs


class LSTMTagger(nn.Module):
    def __init__(self, word_emb_dim=384, lstm_hidden_dim=40, lstm_layers_count=1):
        super().__init__()
        self.word_emb_dim = word_emb_dim

        self.word_emb = ExtendedEmbedding(2, word_emb_dim)
        self.lstm = nn.LSTM(word_emb_dim, lstm_hidden_dim,
                            num_layers=lstm_layers_count,bidirectional=True)
        self.tagger = nn.Linear(2*lstm_hidden_dim, 2)

    def forward(self, inputs):
        embeddings = self.word_emb.embeddings(inputs)
        res, _ = self.lstm(embeddings)
        tag = self.tagger(res)
        return F.softmax(tag, 1)


def do_epoch(model, criterion, data, batch_size, optimizer=None, name=None, top_two=False):
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

                correct_count_delta, first_occurance_of_code_score = model.accuracy(y_batch, logits, indices, top_two)
                correct_count +=  correct_count_delta  * batch_size
                sum_count += batch_size

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}, Previous code mean = {:.2%}'.format(
                    name, epoch_loss / (i+1), correct_count / sum_count, first_occurance_of_code_score)
                )
    #if model.embeddings_size == 320 and not is_train:
    #    unknown_emb_accuracy = zero_embeddings_data_accuracy(model)
    #    print('Unknown embeddings accuracy = {:.2%}'.format(unknown_emb_accuracy))
    return epoch_loss / batches_count, correct_count / sum_count

def bug_probability(y_pred):
    preds = []
    for n in range(y_pred.shape[0]):
        pred = []
        for i in range(y_pred.shape[1]):
            if y_pred[n][i][0] > y_pred[n][i][1]:
                pred.append(0)
            else:
                pred.append(y_pred[n][i][1])
        preds.append(pred)
    return np.array(preds)

def fit(model, criterion, optimizer, train_data, epochs_count=1, batch_size=32,
        val_data=None, val_batch_size=None, scheduler=None, top_two=False):
        
    if not val_data is None and val_batch_size is None:
        val_batch_size = batch_size
    all_train_acc = []
    all_val_acc = []
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, train_acc = model.do_epoch(model, criterion, train_data, batch_size, 
        optimizer, name_prefix + 'Train:', top_two)
        
        if not val_data is None:
            val_loss, val_acc = model.do_epoch(model, criterion, val_data, val_batch_size,
            None, name_prefix + '  Val:', top_two)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        if scheduler:
            scheduler.step()
    return np.array(all_train_acc).max(), np.array(all_val_acc).max()


def zero_embeddings_data_accuracy(model):
    X, y_true, _ = read_data('unknown_embeddings.npy', 'unknown_embeddings_labels.npy')
    model.model.train(False)
    logits = model.model(FloatTensor(X))
    matched_positions = 0
    y_pred = bug_probability(logits)
    for n in range(y_pred.shape[0]):
        if (np.argmax(y_pred[n]) == np.argmax(y_true[n])):
            matched_positions += 1
    return matched_positions / y_pred.shape[0]


class BugLocalizationModel():
    lrs = [1e-2, 1e-3]
    epochs = [20]
    anneal_coefs = [0.25]
    anneal_epochs = [5, 10, 15]
    hidden_dim = [40, 60, 80]
    optims = [optim.Adam]
    def __init__(self, embeddings_path='X.npy', labels_path='y.npy',
                       reports_path='report_ids.npy', ranked=False, flat=False):
        self.X, self.y, self.reports_used = read_data(embeddings_path, labels_path, reports_path)
        self.has_code = self.check_code_embeddings(self.X, self.y)
        #count_embeddings_before_buggy_method(self.has_code)
        self.train_x, self.train_y, self.test_x, self.test_y = train_test_split(self.X, self.y)
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

        if flat:
            (self.train_x, self.train_y, self.test_x, 
            self.test_y, self.report_info) = flat_model.test_train_split_flat(self.X, self.y)
            self.do_epoch = flat_model.do_epoch



    def check_code_embeddings(self, X, y):
        has_code = y.copy()
        samples_count = X.shape[0]
        report_length = X.shape[1]
        embeddings_size = X.shape[2]
        for n in range(samples_count):
            for i in range(report_length):
                has_code[n][i] = 0
                if np.sum(X[n][i]) == X.shape[2]:
                    y[n][i] = 0
                    X[n][i] = np.zeros((embeddings_size,))
                    has_code[n][i] = 2
                elif np.sum(X[n][i]) != 0:
                    has_code[n][i] = 1
        return has_code


    def count_embeddings_before_buggy_method(self, has_code):
        first_occurance_of_code = []
        for n in range(self.y.shape[0]):
            for i in range(self.y.shape[1]):
                if self.has_code[n][i] == 1:
                    first_occurance_of_code.append(i < np.argmax(self.y[n]))
                    break
        percentile = 0
        if first_occurance_of_code != []:
            percentile = np.array(first_occurance_of_code).mean() * 100
        print("{:.2f}% of reports has code embeddings before buggy method".format(percentile))
        

    def create_list_of_train_hyperparameters(self):
        params = it.product(self.lrs, self.epochs, self.optims, self.anneal_coefs,
                            self.anneal_epochs, self.hidden_dim)
        params = map(lambda param: Parameters(*param), params)
        return params


    def accuracy(self, y_true, y_pred, indices=[], top_two=False):
        matched_positions = 0
        first_occurance_of_code = []
        y_pred = bug_probability(y_pred)
        for n in range(y_pred.shape[0]):
            max_preds = y_pred[n].argsort()[-2:][::-1]
            if (max_preds[0] == np.argmax(y_true[n])) or (top_two and np.argmax(y_true[n]) in list(max_preds)):
                matched_positions += 1
                if np.array(indices).size > 0:
                    indice = indices[n]
                    for i in range(y_pred.shape[1]):
                        if self.has_code[indice][i] == 1:
                            first_occurance_of_code.append(i < np.argmax(y_true[n]))
                            break

        first_occurance_of_code_score = 0
        if first_occurance_of_code != []:
            first_occurance_of_code_score = np.array(first_occurance_of_code).mean()

        return matched_positions/y_true.shape[0], first_occurance_of_code_score
    
    
    def train(self, params, model_to_train, top_two=False):
        self.run_records = []
        self.params = list(self.create_list_of_train_hyperparameters())
        for param in self.params:
            print(param)
            loss = self.criterion()
            self.model = model_to_train(
                word_emb_dim = self.embeddings_size,
                lstm_hidden_dim=param.dim)

            optimizer = param.optim(self.model.parameters(), lr=param.lr)
            scheduler = StepLR(optimizer, step_size = param.anneal_epoch, gamma = param.anneal_coef)
            train_acc, val_acc = fit(self, loss, optimizer, 
                                     train_data=(self.train_x, self.train_y), epochs_count=param.epoch,
                                     batch_size=128, val_data=(self.test_x, self.test_y),
                                     val_batch_size=128, scheduler=scheduler, top_two=top_two)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            self.run_records.append(RunResult(train_acc, val_acc))
        return self.run_records

    
    def save_results(self, name):
        results = []
        for i, record in enumerate(self.run_records):
            result = []
            result = {'train_acc':record.train_history, 'val_acc':record.val_history}
            param = self.params[i]._asdict()
            param = dict(map(lambda item: (str(item[0]), str(item[1])), param.items()))
            result.update(param)
            results.append(result)
        results.sort(key=lambda x: -x['val_acc'])
        f = open('results' + name + '.txt', 'w')
        json.dump(results, f, indent=4)
        f.close()
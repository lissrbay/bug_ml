import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
import math
from tqdm import tqdm
import model as base_model_functions
from collections import defaultdict


def to_flat(X, y, test=False):
    X_flat = X.reshape((-1, X.shape[2]))
    y_flat = y.flatten()
    report_info = []
    for n in range(X.shape[0]):
        for i in range(X.shape[1]):
            if np.sum(X[n][i]) == 0.0 or np.sum(X[n][i]) == 320:
                y_flat[n*X.shape[1]+ i] = 3
            
            report_info.append([n, i])

    X_flat = X_flat[~(y_flat == 3)]
    report_info = np.array(report_info)[~(y_flat == 3)]
    y_flat = y_flat[~(y_flat == 3)]
    if not test:
        X_flat_zeros = X_flat[~(y_flat==1)]
        y_flat_zeros = y_flat[~(y_flat==1)]
        X_flat_ones = X_flat[(y_flat==1)]
        y_flat_ones = y_flat[(y_flat==1)]
        n_samples = X_flat_ones.shape[0]
        X_flat_zeros = X_flat_zeros[:2*n_samples]
        y_flat_zeros = y_flat_zeros[:2*n_samples]
        X_flat = np.concatenate([X_flat_zeros, X_flat_ones])
        y_flat = np.concatenate([y_flat_zeros, y_flat_ones])

    return X_flat, y_flat, report_info


def test_train_split_flat(X, y):
    train_x, train_y, test_x, test_y = base_model_functions.train_test_split(X, y)
    train_x, train_y, _ = to_flat(train_x, train_y)
    test_x, test_y, report_info = to_flat(test_x, test_y, True)
    return train_x, train_y, test_x, test_y, report_info


class SimpleModel(nn.Module):
    def __init__(self, word_emb_dim=384,lstm_hidden_dim=0, drop=0.2):
        super().__init__()
        self.tagger = nn.Sequential(
            nn.Linear(word_emb_dim, 100),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2), 
       )
        
    def forward(self, inputs):
        tag = self.tagger(inputs)
        return F.softmax(tag, 1)


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for n in range(y_pred.shape[0]):
        if (np.argmax(y_pred[n].detach().numpy()) == 1 and y_true[n] == 1):
            tp += 1
        elif (np.argmax(y_pred[n].detach().numpy()) == 0 and y_true[n] == 0):
            tn += 1
        elif (np.argmax(y_pred[n].detach().numpy()) == 0 and y_true[n] == 1):
            fn += 1
        elif (np.argmax(y_pred[n].detach().numpy()) == 1 and y_true[n] == 0):
            fp += 1

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fraction_ones = tp / y_true.detach().numpy().sum() if y_true.sum() > 0 else 0.0
    fraction_zeros = tn / (y_true.shape[0] - y_true.detach().numpy().sum()) if y_true.sum() > 0 else 0.0

    return (sensitivity + specificity) / 2, fraction_ones, fraction_zeros


def accuracy_val(y_true, y_pred, report_info):
    tp, tn, fp, fn = 0, 0, 0, 0
    for n in range(y_pred.shape[0]):
        if (np.argmax(y_pred[n].detach().numpy()) == y_true[n] and y_true[n] == 1):
            tp += 1
        if (np.argmax(y_pred[n].detach().numpy()) == y_true[n] and y_true[n] == 0):
            tn += 1
        if (np.argmax(y_pred[n].detach().numpy()) == 0 and y_true[n] == 1):
            fn += 1
        if (np.argmax(y_pred[n].detach().numpy()) == 1 and y_true[n] == 0):
            fp += 1
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
    fraction = tp / y_true.detach().numpy().sum() 

    report_scores = defaultdict(list)
    for n in range(y_pred.shape[0]):
        report_id, pos = report_info[n]
        if np.argmax(y_pred[n].detach().numpy()) == 0:
            score = 0
        else:
            score = np.max(y_pred[n].detach().numpy())
            
        report_scores[report_id].append((score, pos, y_true[n]))
    matched_reports = 0
    for n, scores in report_scores.items():
        max_score_pos = np.argmax([score[0] for score in scores])
        if scores[max_score_pos][2] == 1:
            matched_reports+=1

    return (sensitivity + specificity) / 2, fraction, matched_reports / len(report_scores)


def do_epoch(model, criterion, data, batch_size, optimizer=None, name=None):
    epoch_loss = 0
    correct_count_balansed = 0
    sum_count_balansed = 0
    correct_count_fraction_ones = 0
    sum_count_fraction_ones = 0
    correct_count_fraction_zeros = 0
    sum_count_fraction_zeros = 0
    is_train = not optimizer is None
    name = name or ''
    model.model.train(is_train)
    
    batches_count = math.ceil(len(data[0]) / batch_size)
    if not is_train:
        X, y = data
        X_batch, y_batch = FloatTensor(X), LongTensor(y)
        logits = model.model(X_batch)
        loss = criterion(logits, y_batch)
        epoch_loss += loss.item()

        balansed_acc, fraction_acc, report_acc = accuracy_val(y_batch, logits, model.report_info)
        correct_count_balansed += balansed_acc
        print('{:>5s} Loss = {:.5f}, Balansed accuracy = {:.2%}, Fraction  accuracy = {:.2%}, Report accuracy = {:.2%}'.format(
            name, epoch_loss, correct_count_balansed ,
            fraction_acc , report_acc)
        )
        return epoch_loss ,(correct_count_balansed, fraction_acc)
    
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, (X_batch, y_batch, _) in enumerate(base_model_functions.iterate_batches(data, batch_size)):
                X_batch, y_batch = FloatTensor(X_batch), LongTensor(y_batch)
                logits = model.model(X_batch)
                loss = criterion(logits, y_batch)
                epoch_loss += loss.item()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                balansed_acc, fraction_ones_acc, fraction_zeros_acc = accuracy(y_batch, logits)
                correct_count_balansed += balansed_acc * batch_size
                sum_count_balansed += batch_size
                correct_count_fraction_ones += fraction_ones_acc
                correct_count_fraction_zeros += fraction_zeros_acc
                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, Balansed accuracy = {:.2%}, Fraction 1 accuracy = {:.2%}, Fraction 0 accuracy = {:.2%}'.format(
                    name, epoch_loss/(i + 1), correct_count_balansed / sum_count_balansed,
                    correct_count_fraction_ones / (i+1),
                    correct_count_fraction_zeros/(i+1))
                )

    return epoch_loss / batches_count, (correct_count_balansed / sum_count_balansed,
                   correct_count_fraction_ones / batches_count,
                   correct_count_fraction_zeros/batches_count)
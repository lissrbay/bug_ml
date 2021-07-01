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
from . import model as base_model_functions

def rank_prediction(y_pred):
    pred = []
    for i in range(y_pred.shape[1]):
        if y_pred[i][0] > y_pred[i][1]:
            pred.append(0)
        else:
            pred.append(y_pred[i][1])
    bag = np.argmax(pred)
    pred = np.zeros((y_pred.shape[0], 1))
    pred[bag][0] = 1
    pred = FloatTensor(pred)
    pred.requires_grad=True
    return pred


def rank_loss(criterion, logits, y):
    loss = 0

    for n in range(logits.shape[0]):
        sample = logits[n]
        sample = rank_prediction(sample)
        targets = y[n]
        targets[targets == 0] = -1
        neg_samples = sample[targets == -1]
        pos_samples = torch.cat(neg_samples.shape[0]*[sample[targets == 1]])
        targets = targets.reshape(80, 1)
        loss += criterion(pos_samples, neg_samples, torch.ones(targets.shape[0]))

    return loss

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
            for i, (X_batch, y_batch, indices) in enumerate(base_model_functions.iterate_batches(data, batch_size)):
                X_batch, y_batch = FloatTensor(X_batch), LongTensor(y_batch)
                logits = model.model(X_batch)

                #loss = criterion(logits.transpose(2, 1), y_batch)
                loss = 0
                loss += rank_loss(criterion, logits, y_batch)
                epoch_loss += loss.item()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                correct_count_delta, first_occurance_of_code_score = model.accuracy(y_batch, logits, indices)
                correct_count +=  correct_count_delta  * batch_size
                sum_count += batch_size

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}, Previous code mean = {:.2%}'.format(
                    name, epoch_loss / (i+1), correct_count / sum_count, first_occurance_of_code_score)
                )


    return epoch_loss / batches_count, correct_count / sum_count
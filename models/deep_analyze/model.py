from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from modules import DeepAnalyzeAttention

from torchmetrics import Metric, MetricCollection

from torchcrf import CRF


def precision(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(pred.sum().item(), 1)


def recall(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(labels.sum().item(), 1)


def metrics_to_str(metrics: Dict[str, torch.Tensor]):
    return " ".join(f"{k}: {v.item():.3f}" for k, v in metrics.items())


class Precision(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):

        self.tp += torch.sum(preds[mask] * target[mask], dim=0)
        self.total += max(target[mask].sum().item(), 1)

    def compute(self):
        return round((self.tp.float() / self.total).item(), 3)


class Recall(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):

        self.tp += torch.sum(preds[mask] * target[mask], dim=0)
        self.total += max(preds[mask].sum().item(), 1)

    def compute(self):
        return round((self.tp.float() / self.total).item(), 3)


class DeepAnalyze(pl.LightningModule):
    def __init__(self, feature_size, lstm_hidden_size, lstm_num_layers, n_tags):
        super().__init__()
        self.padding = 0
        self.bi_listm = nn.LSTM(feature_size, lstm_hidden_size,
                                num_layers=lstm_num_layers, bidirectional=True)
        self.attention = DeepAnalyzeAttention(lstm_hidden_size * 2, n_tags)
        self.crf = CRF(n_tags)
        self.lstm_dropout = nn.Dropout(0.25)

        self.train_metrics = MetricCollection([Precision(), Recall()])
        self.val_metrics = MetricCollection([Precision(), Recall()])

    def forward(self, inputs, mask):
        seq_len, batch_size = mask.shape

        x, _ = self.bi_listm(inputs)
        x = self.lstm_dropout(x)
        x = self.attention(x, mask)
        preds = self.crf.decode(x, mask)

        preds = [pred + [0] * (seq_len - len(pred)) for pred in preds]
        preds = torch.tensor(preds).transpose(0, 1).to(inputs.device)

        return preds

    def training_step(self, batch, batch_idx):
        inputs, labels, mask = batch
        x, _ = self.bi_listm(inputs)
        x = self.lstm_dropout(x)
        x = self.attention(x, mask)

        loss = -self.crf(x, labels, mask)

        with torch.no_grad():
            preds = self.forward(inputs, mask)

        self.train_metrics.update(preds, labels, mask)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args):
        inputs, labels, mask = batch
        x, _ = self.bi_listm(inputs)
        x = self.lstm_dropout(x)
        x = self.attention(x, mask)
        loss = -self.crf(x, labels, mask)
        with torch.no_grad():
            preds = self.forward(inputs, mask)

        self.val_metrics.update(preds, labels, mask)
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        self.log("val_metrics", self.val_metrics.compute())
        self.val_metrics.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        self.log("train_metrics", self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

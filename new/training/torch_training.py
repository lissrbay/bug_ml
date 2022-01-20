from pathlib import Path
from typing import List, Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchmetrics import MetricCollection

from new.data.report import Report
from new.model.lstm_tagger import LstmTagger
from new.training.data import ReportsDataModule
from new.training.metrics import Precision, Recall, TopkAccuracy


class TrainingModule(pl.LightningModule):
    def __init__(self, tagger: LstmTagger):
        super().__init__()
        self.tagger = tagger

        self.train_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(3)])
        self.val_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(3)])

    def training_step(self, batch, batch_idx):
        reports, target, masks = batch
        mask = torch.cat(masks, dim=1)

        if self.tagger.with_crf:
            emissions = torch.cat([self.tagger.calc_emissions(report, mask) for report, mask in zip(reports, masks)],
                                  dim=1)
            loss = -self.tagger.crf(emissions, target, mask)
        else:
            scores = self.tagger.forward(reports, masks)
            loss = nn.functional.cross_entropy(scores.transpose(1, 2), target, ignore_index=2)

        with torch.no_grad():
            scores = self.tagger.forward(reports, masks)
            preds = scores.argmax(dim=-1)

        self.train_metrics.update(preds, target, mask, scores=scores)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args):
        reports, target, masks = batch
        mask = torch.cat(masks, dim=1)

        if self.tagger.with_crf:
            emissions = torch.cat([self.tagger.calc_emissions(report, mask) for report, mask in zip(reports, masks)],
                                  dim=1)
            loss = -self.tagger.crf(emissions, target, mask)
        else:
            scores = self.tagger.forward(reports, masks)
            loss = nn.functional.cross_entropy(scores.transpose(1, 2), target, ignore_index=2)

        with torch.no_grad():
            scores = self.tagger.forward(reports, masks)
            preds = scores.argmax(dim=-1)

        self.val_metrics.update(preds, target, mask, scores=scores)

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


def train_lstm_tagger(tagger: LstmTagger, reports: List[Report], target: List[List[int]],
                      cached_dataset_path: Optional[Path] = None) -> LstmTagger:
    datamodule = ReportsDataModule(reports, target, 4, 80)
    model = TrainingModule(tagger)

    trainer = Trainer(gpus=1)
    trainer.fit(model, datamodule)

    return tagger

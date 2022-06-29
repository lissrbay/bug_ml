import gc
from typing import List, Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchmetrics import MetricCollection
# from transformers import Trainer, TrainingArguments

from new.data.report import Report
from new.model.lstm_tagger import LstmTagger
from new.training.data import ReportsDataModule
from new.training.metrics import Precision, Recall, TopkAccuracy

from commode_utils.losses import SequenceCrossEntropyLoss

class TrainingModule(pl.LightningModule):
    def __init__(self, tagger: LstmTagger):
        super().__init__()
        self.tagger = tagger

        self.train_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(1)])
        self.val_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(1)])

        self.softmax = torch.nn.Softmax(dim=-1)
        self.celoss = SequenceCrossEntropyLoss(reduction="batch-mean", pad_idx=2)

    def training_step(self, batch, batch_idx):
        reports, target, masks = batch
        mask = torch.cat(masks, dim=1)

        if self.tagger.with_crf:
            emissions = torch.cat([self.tagger.calc_emissions(report, mask) for report, mask in zip(reports, masks)],
                                  dim=1)
            loss = -self.tagger.crf(emissions, target, mask)
        else:
            scores = self.tagger.forward(reports, masks)
            loss = self.celoss(scores, target)

        with torch.no_grad():
            scores = self.tagger.forward(reports, masks)
            preds = scores.argmax(dim=-1)

        scores = self.softmax(scores)
        self.train_metrics.update(preds, target, mask, scores=scores)

        self.log("train_loss", loss)

        with open('grad.log', "a") as f:
            for name, para in self.tagger.report_encoder.model.named_parameters():
                if 'layer.11.output.dense.weight' in name:
                    # print(str())
                    f.write(str(para))
            f.write("\n\n")


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
            loss = self.celoss(scores, target)

        with torch.no_grad():
            scores = self.tagger.forward(reports, masks)
            preds = scores.argmax(dim=-1)

        scores = self.softmax(scores)
        self.val_metrics.update(preds, target, mask, scores=scores)

        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        self.log("val_metrics", self.val_metrics.compute())
        print(self.val_metrics.compute())
        self.val_metrics.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        self.log("train_metrics", self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)


def train_lstm_tagger(tagger: LstmTagger, reports: List[Report], target: List[List[int]], batch_size: int,
                      max_len: int, cpkt_path: Optional[str] = None) -> LstmTagger:
    datamodule = ReportsDataModule(reports, target, batch_size, max_len)
    model = TrainingModule(tagger)
    if cpkt_path:
        state_dict = torch.load(cpkt_path)["state_dict"]
        model.load_state_dict(state_dict)
        for param in model.tagger.report_encoder.model.parameters():
            param.requires_grad = False
        for name, param in model.tagger.report_encoder.model.named_parameters():
            # if "11" in name or "10" in name:
            param.requires_grad = True

        with open('grad.log', "w") as f:
            # for name, para in model.tagger.report_encoder.model.named_parameters():
            #     if 'layer.11.output.LayerNorm.weight' in name:
            #         f.write(para.requires_grad)
            #         f.write(para)
            f.write("")
    model.tagger.report_encoder.model.gradient_checkpointing_enable()

    trainer = Trainer(gpus=1)
    trainer.validate(model, datamodule)
    trainer.fit(model, datamodule)


    return tagger

from typing import List, Any, Optional

import pytorch_lightning as pl
import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torchmetrics import MetricCollection
from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.concat_encoders import ConcatReportEncoders

from new.data.report import Report
from new.model.lstm_tagger import LstmTagger
from new.training.data import ReportsDataModule
from new.training.metrics import Precision, Recall, TopkAccuracy, BootStrapper

from pytorch_lightning import loggers as pl_loggers


class TrainingModule(pl.LightningModule):
    def __init__(self, tagger: LstmTagger, lr: float = 1e-5):
        super().__init__()
        self.tagger = tagger
        bootstrap_func = lambda x: BootStrapper(x, num_bootstraps=100, prefix=x.name, quantile=0.01)
        bootstrap_prefixes = ['Precision', 'Recall', 'TopkAccuracy', 'TopkAccuracy3', 'TopkAccuracy5', 'TopkAccuracy_monitor']
        self.train_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])) + [TopkAccuracy(1)])), prefix="train/")
        self.val_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])) + [TopkAccuracy(1)])), prefix="val/")
        self.test_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])) + [TopkAccuracy(1)])), prefix="test/")

        self.softmax = torch.nn.Softmax(dim=-1)
        self.mseloss = torch.nn.MSELoss()
        self.celoss = SequenceCrossEntropyLoss(reduction="batch-mean")
        self.lr = lr

    def build_scaffle_labels(self, reports: List[Report], target: torch.Tensor):
        max_len = target.shape[0]
        labels = [
            [frame.meta["ground_truth"] for frame in report.frames][:max_len] for report in reports
        ]
        labels = [label + [0] * (max_len - len(label)) for label in labels]

        labels_tensor = torch.LongTensor(labels)

        return labels_tensor.permute(1, 0).to(self.device)

    def calculate_step(self, batch, metrics):
        reports, target, masks = batch
        mask = torch.cat(masks, dim=1)

        if self.tagger.with_crf:
            emissions = torch.cat([self.tagger.calc_emissions(report, mask) for report, mask in zip(reports, masks)],
                                  dim=1)
            loss = -self.tagger.crf(emissions, target, mask)
        else:
            scores = self.tagger.forward(reports, masks)
            if self.tagger.scaffle:
                loss = self.mseloss(scores, target.float())
            else:
                loss = self.celoss(scores, target)

        with torch.no_grad():
            scores = self.tagger.forward(reports, masks)
            if self.tagger.scaffle:
                preds = (scores > 0.5).int()
            else:
                preds = scores.argmax(dim=-1)

        if self.tagger.scaffle:
            scores = scores.unsqueeze(-1)
            scores = torch.cat((1 - scores, scores), dim=-1)
        else:
            scores = self.softmax(scores)

        if self.tagger.scaffle:
            target = self.build_scaffle_labels(reports, target)

        metrics.update(preds, target, mask, scores=scores)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_step(batch, self.train_metrics)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args):
        loss = self.calculate_step(batch, self.val_metrics)
        return loss

    def test_step(self, batch, *args):
        loss = self.calculate_step(batch, self.test_metrics)
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        metrics_dict = self.val_metrics.compute()
        self.log_dict(metrics_dict)
        print(metrics_dict)
        self.val_metrics.reset()

    def test_epoch_end(self, outputs: List[Any]) -> None:
        super().test_epoch_end(outputs)
        metrics_dict = self.test_metrics.compute()
        self.log_dict(metrics_dict)
        print(metrics_dict)
        self.test_metrics.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def train_lstm_tagger(tagger: LstmTagger, reports: List[Report], target: List[List[int]], batch_size: int,
                      max_len: int, model_name: Optional[str], lr: float, caching: bool = False,
                      cpkt_path: Optional[str] = None, device: str = None, max_epoch: int = 20) -> LstmTagger:
    datamodule = ReportsDataModule(reports, target, batch_size, max_len, model_name)
    model = TrainingModule(tagger, lr)

    if cpkt_path:
        state_dict = torch.load(cpkt_path)["state_dict"]
        model.load_state_dict(state_dict)
        if type(model.tagger.report_encoder) == ConcatReportEncoders:
            model_parameters = model.tagger.report_encoder.report_encoders[0].model.parameters()
        else:
            model_parameters = model.tagger.report_encoder.model.parameters()

        for param in model_parameters:
            param.requires_grad = True

    gpus = 1 if device == "cuda" else None

    callbacks = [ModelCheckpoint(
        monitor="val/TopkAccuracy_monitor",
        mode="max"
    )]

    if model_name == "bert" and not caching:
        if type(model.tagger.report_encoder) == ConcatReportEncoders:
            model.tagger.report_encoder.report_encoders[0].model.gradient_checkpointing_enable()
        else:
            model.tagger.report_encoder.model.gradient_checkpointing_enable()

        callbacks.append(ZeroCallback())

    logs_name = model_name
    if caching:
        logs_name += "_caching"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=logs_name)

    trainer = Trainer(gpus=gpus, callbacks=callbacks, deterministic=True, logger=tb_logger, max_epochs=max_epoch)

    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")
    return tagger


class ZeroCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        layer_names = ["11", "10"]
        if type(pl_module.tagger.report_encoder) == ConcatReportEncoders:
            parameters = pl_module.tagger.report_encoder.report_encoders[0].model.named_parameters()
        else:
            parameters = pl_module.tagger.report_encoder.model.named_parameters()
        for name, param in parameters:
            if all(x not in name for x in layer_names):
                if param.grad is not None:
                    param.grad *= 0

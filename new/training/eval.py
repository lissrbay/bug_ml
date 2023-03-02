import os

import torch
from torchmetrics import MetricCollection
from typing import List, Any, Optional
from model.lstm_tagger import LstmTagger
from new.training.data import ReportsDataModule
import argparse
import glob
import json

from training.torch_training import TrainingModule
from training.train import read_reports, init_model
from pytorch_lightning import loggers as pl_loggers, Trainer
import pytorch_lightning as pl
from new.training.metrics import Precision, Recall, TopkAccuracy


class EvalModule(pl.LightningModule):
    def __init__(self, tagger: LstmTagger, lr: float = 1e-5):
        super().__init__()
        self.tagger = tagger
        self.val_metrics = MetricCollection([Precision(), Recall(),
                                             TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)],
                                            prefix="val/")
        self.test_metrics = MetricCollection([Precision(), Recall(),
                                              TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)],
                                             prefix="test/")

    def validation_step(self, batch, *args):
        loss = self.calculate_step(batch, self.val_metrics)
        return loss

    def test_step(self, batch, *args):
        loss = self.calculate_step(batch, self.test_metrics)
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        self.log_dict(self.val_metrics.compute())
        print(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_epoch_end(self, outputs: List[Any]) -> None:
        super().test_epoch_end(outputs)
        self.log_dict(self.test_metrics.compute())
        print(self.test_metrics.compute())
        self.test_metrics.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()


def eval_models(reports_path, logdir, config_path):
    for dir in glob.glob(logdir + "/*"):
        last_model_run = sorted(list(glob.glob(dir + "/*")))[-1]
        model_name = dir.split("/")[-1]
        reports, target = read_reports(reports_path, model_name)
        with open(config_path, "r") as f:
            config_name = model_name.replace("_caching", "")
            config = json.load(f)
            train_params = config[config_name]["training"]

        datamodule = ReportsDataModule(reports, target, train_params['batch_size'], train_params['max_len'], model_name)
        logs_name = model_name
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="/Users/e.poslovskaya/bug_ml_copy_2/bug_ml_copy_2/new/training/lightning_logs_test/", name=logs_name)
        gpus = None

        caching = "caching" in model_name
        tagger = init_model(config_name, config, caching, reports, target)

        model = TrainingModule(tagger)
        cpkt_path = list(glob.glob(os.path.join(last_model_run, "checkpoints", "*")))[-1]
        state_dict = torch.load(cpkt_path, map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
        trainer = Trainer(gpus=gpus, callbacks=None, deterministic=True, logger=tb_logger, max_epochs=1)
        trainer.test(model, datamodule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--config_path", type=str, default="/Users/e.poslovskaya/bug_ml_copy_2/bug_ml_copy_2/new/training/config.json")

    args = parser.parse_args()

    eval_models(args.reports_path, args.logdir, args.config_path)


if __name__ == '__main__':
    main()
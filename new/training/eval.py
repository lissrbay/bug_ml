import os
import sys
from collections import defaultdict

sys.path.insert(0, './../../')
import torch
from torchmetrics import MetricCollection
from typing import List, Any, Optional
from new.model.lstm_tagger import LstmTagger
from new.training.data import ReportsDataModule
import argparse
import glob
import json
import numpy as np
from new.training.torch_training import TrainingModule
from new.training.train import read_reports, init_model
from pytorch_lightning import loggers as pl_loggers, Trainer
import pytorch_lightning as pl
from new.training.metrics import Precision, Recall, TopkAccuracy


class EvalModule(TrainingModule):
    def __init__(self, tagger: LstmTagger, lr: float = 1e-5):
        super().__init__()
        self.tagger = tagger
        bootstrap_func = lambda x: BootStrapper(x, num_bootstraps=100, prefix=x.name, quantile=0.01)
        bootstrap_prefixes = ['Precision', 'Recall', 'TopkAccuracy', 'TopkAccuracy3', 'TopkAccuracy5']
        self.train_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])))), prefix="train/")
        self.val_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])))), prefix="val/")
        self.test_metrics = MetricCollection(dict(zip(bootstrap_prefixes, list(map(bootstrap_func, 
        [Precision(), Recall(), TopkAccuracy(1), TopkAccuracy(3), TopkAccuracy(5)])))), prefix="test/")

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


def eval_models(reports_path, logdir, config_path, is_test=False):
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
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="./bs_logs_private/", name=logs_name)
        gpus = 1

        caching = "caching" in model_name
        tagger = init_model(config_name, config, caching, reports, target)
        model = TrainingModule(tagger, logs_save_path = "./bs_logs_private/"  + model_name + '/' + last_model_run.split('/')[-1] + '/')
        cpkt_path = list(glob.glob(os.path.join(last_model_run, "checkpoints", "*")))[-1]
        state_dict = torch.load(cpkt_path, map_location=torch.device('cuda:0'))["state_dict"]
        model.load_state_dict(state_dict)
        if is_test:
            trainer = Trainer(gpus=gpus, callbacks=None, deterministic=True, logger=tb_logger, max_epochs=1)
            trainer.test(model, datamodule)
        else:
            for report_batch in datamodule.test_dataloader():
                preds = tagger.predict(report_batch)
                torch.save(preds, model_name + '_preds')

def eval_diff_predictions(reports_path, logdir, config_path):
    pass

def eval_baseline(reports_path):
    reports, target = read_reports(reports_path, "baseline")
    datamodule = ReportsDataModule(reports, target, batch_size=64, max_len=80, label_style="baseline")
    datamodule.setup()
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="./lightning_logs_test/", name=reports_path.replace('/', '_').replace('.', '_'))
    acc = {'accuracy@1': 0, 'accuracy@3': 0, 'accuracy@5': 0}
    preds = []
    for reports_batch, targets_batch, _ in datamodule.test_dataloader():
        targets_batch = targets_batch.T
        preds.append(targets_batch)
    preds = torch.cat(preds, dim=0)
    bd_metrics = defaultdict(list)
    for _ in range(100):
        bootstraped_ids = np.random.choice(preds.shape[0], preds.shape[0], replace=True)
        targets_batch = preds[bootstraped_ids, :]
        acc = {'accuracy@1': 0, 'accuracy@3': 0, 'accuracy@5': 0}
        c = 0

        for i in acc.keys():
            top_k = int(i.split("@")[1])
            for j, t in enumerate(targets_batch):
                #r_t = [frame.meta['label'] for frame in r.frames]
                #print(t.shape)
                if sum(t[:top_k]) > 0:
                    acc[i] += 1#int(sum(t[:top_k]))
        c += len(preds)
        for k in acc.keys():
            bd_metrics[k].append(acc[k] / c)

    for k in acc.keys():
        print("Mean:", np.mean(bd_metrics[k]))
        print("Quantile 0.005:", np.quantile(bd_metrics[k], 0.01/2))
        print("Quantile 0.995:", np.quantile(bd_metrics[k], (1-0.01/2)))

    tb_logger.log_metrics(acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--config_path", type=str, default="/home/centos/bug_ml/new/training/config.json")

    args = parser.parse_args()
    eval_baseline(args.reports_path)
    eval_models(args.reports_path, args.logdir, args.config_path)


if __name__ == '__main__':
    main()
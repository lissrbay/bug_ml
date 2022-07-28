import argparse
import json
import logging
import os
import random
import sys
from typing import List, Optional, cast

import numpy.random
import torch
from code2seq.model import Code2Seq
from omegaconf import DictConfig, OmegaConf

from new.data.labeled_path_context_storage import LabeledPathContextStorage
from new.data.report import Report
from new.data_aggregation.utils import iterate_reports
from new.model.lstm_tagger import LstmTagger
from new.model.report_encoders.code2seq_report_encoder import Code2SeqReportEncoder
from new.model.report_encoders.codebert_encoder import RobertaReportEncoder
from new.model.report_encoders.scaffle_report_encoder import ScaffleReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder
from new.training.torch_training import train_lstm_tagger

sys.path.insert(0, './../../')

loger = logging.getLogger('lightning')
loger.info(...)
loger.debug(...)


def make_target(reports: List[Report], label_style: Optional[str]) -> List[List[int]]:
    targets = []
    for report in reports:
        if label_style == "scaffle":
            target = [frame.meta["label"] for frame in report.frames]
        else:
            target = [frame.meta["ground_truth"] for frame in report.frames]
        targets.append(target)
    return targets


def train(reports_path: str, save_path: str, model_name: Optional[str], caching: bool = False,
          checkpoint_path: Optional[str] = None):
    seed = 9219321

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    reports = []
    for file_name in iterate_reports(reports_path):
        report_path = os.path.join(reports_path, file_name)
        report = Report.load_report(report_path)
        if report.frames:
            if sum(frame.meta["label"] for frame in report.frames) > 0:
                reports.append(report)

    reports = reports

    target = make_target(reports, model_name)

    with open("config.json", "r") as f:
        config = json.load(f)

    model_names = ["scaffle", "deep_analyze", "code2seq"]

    if model_name is not None:
        if model_name == "scaffle":
            encoder = ScaffleReportEncoder(**config["models"]["scaffle"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                scaffle=True,
                **config["models"]["scaffle"]["tagger"]
            )
            config["training"]["lr"] = config["models"]["scaffle"]["lr"]
        elif model_name == "deep_analyze":
            encoder = TfIdfReportEncoder(**config["models"]["deep_analyze"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                **config["models"]["deep_analyze"]["tagger"]
            )
            config["training"]["lr"] = config["models"]["deep_analyze"]["lr"]
        elif model_name == "code2seq":
            config_path = config["code2seq_config_path"]
            cli_path = config["astminer_config_path"]
            ast_config_path = config["astminer_config_path"]

            __config = cast(DictConfig, OmegaConf.load(config_path))

            code2seq = Code2Seq.load_from_checkpoint(__config.checkpoint, map_location=torch.device("cpu"))

            storage = LabeledPathContextStorage(cli_path, ast_config_path, code2seq.vocabulary, __config,
                                                **config["code2seq_storage"])

            storage.load_data(reports, mine_files=False, process_mined=False, remove_all=False)

            encoder = Code2SeqReportEncoder(code2seq, storage)

            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                layers_num=2,
                hidden_dim=200
            )

        else:
            raise ValueError(f"Wrong model type. Should be in {model_names}")
    else:
        # pass
        encoder = RobertaReportEncoder(caching=caching, frames_count=config["training"]["max_len"], device='cuda')
        if caching:
            for param in encoder.model.parameters():
                param.requires_grad = False

        # path_to_precomputed_embs="/home/lissrbay/Загрузки/code2seq_embs"),
        # GitFeaturesTransformer(
        #    frames_count=config["training"]["max_len"]).fit(reports, target),
        # MetadataFeaturesTransformer(frames_count=config["training"]["max_len"])

        tagger = LstmTagger(
            encoder,
            max_len=config["training"]["max_len"],
            layers_num=2,
            hidden_dim=250
        )

    tagger = train_lstm_tagger(tagger, reports, target, caching=caching, label_style=model_name,
                               cpkt_path=checkpoint_path, **config["training"])

    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    train(args.reports_path, args.save_path, "scaffle")
    # train(args.reports_path, args.save_path, None, caching=True)
    # train(args.reports_path, args.save_path, None, checkpoint_path="/home/dumtrii/Documents/practos/spring2/bug_ml/new/training/lightning_logs/version_368/checkpoints/epoch=30-step=18909.ckpt")


if __name__ == '__main__':
    main()

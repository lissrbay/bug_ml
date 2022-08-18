import argparse
import json
import logging
import os
import random
import sys
from typing import List, Optional

import numpy.random
import torch

from new.data.report import Report
from new.data_aggregation.utils import iterate_reports
from new.model.lstm_tagger import LstmTagger
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


def train(reports_path: str, config_path: str, model_name: Optional[str], caching: bool = False,
          checkpoint_path: Optional[str] = None):
    print(f"Model name: {model_name}")
    seed = 9219321

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    reports = []
    for file_name in iterate_reports(reports_path):
        report_path = os.path.join(reports_path, file_name)
        report = Report.load_report(report_path)
        if report.frames:
            if sum(frame.meta["label"] for frame in report.frames) > 0:
                reports.append(report)

    reports = sorted(reports, key=lambda r: r.id)

    target = make_target(reports, model_name)

    with open(config_path, "r") as f:
        config = json.load(f)

    max_len = config[model_name]["training"]["max_len"]
    if model_name == "scaffle":
        encoder = ScaffleReportEncoder(**config[model_name]["encoder"]).fit(reports, target)
    elif model_name == "deep_analyze":
        encoder = TfIdfReportEncoder(**config[model_name]["encoder"]).fit(reports, target)
    elif model_name == "bert":
        encoder = RobertaReportEncoder(frames_count=max_len, caching=caching, device=device)
        if caching:
            for param in encoder.model.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Wrong model type")

    # path_to_precomputed_embs="/home/lissrbay/Загрузки/code2seq_embs"),
    # GitFeaturesTransformer(
    #    frames_count=config["training"]["max_len"]).fit(reports, target),
    # MetadataFeaturesTransformer(frames_count=config["training"]["max_len"])

    tagger = LstmTagger(
        encoder,
        max_len=max_len,
        scaffle=model_name == "scaffle",
        device=device,
        **config[model_name]["tagger"]
    )

    tagger = train_lstm_tagger(tagger, reports, target, caching=caching, label_style=model_name,
                               cpkt_path=checkpoint_path, **config[model_name]["training"])

    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--model", type=str, required=True, choices=["scaffle", "deep_analyze", "bert"])
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    train(args.reports_path, args.config_path, args.model, caching=args.caching, checkpoint_path=args.checkpoint_path)
    # train(args.reports_path, args.save_path, args.model, caching=True)
    # train(args.reports_path, args.save_path, args.model, checkpoint_path=args.checkpoint_path"/home/dumtrii/Documents/practos/spring2/bug_ml/new/training/lightning_logs/version_379/checkpoints/epoch=6-step=4836.ckpt")


if __name__ == '__main__':
    main()

# python train.py --reports_path "/Users/Aleksandr.Khvorov/jb/exception-analyzer/data/scaffle_reports"
# python -m new.training.train --reports_path "/home/ubuntu/akhvorov/bugloc/data/scaffle_reports" --config_path "new/training/config.json" --model deep_analyze
# python -m new.training.train --reports_path "/home/ubuntu/akhvorov/bugloc/data/scaffle_reports" --config_path "new/training/config.json" --model bert --caching
# python -m new.training.train --reports_path "/home/ubuntu/akhvorov/bugloc/data/scaffle_reports" --config_path "new/training/config.json" --model bert --caching

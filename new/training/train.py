import argparse
import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import List, Tuple, Optional, cast

import torch
from code2seq.model import Code2Seq
from omegaconf import DictConfig, OmegaConf

from new.data.labeled_path_context_storage import LabeledPathContextStorage
from new.data.report import Report
from new.data_aggregation.utils import iterate_reports
from new.model.lstm_tagger import LstmTagger
from new.model.report_encoders.code2seq_report_encoder import Code2SeqReportEncoder
from new.model.report_encoders.scuffle_report_encoder import ScuffleReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder
from new.training.torch_training import train_lstm_tagger


def read_reports(reports_path: str) -> Tuple[List[Report], List[List[int]]]:
    reports, targets = [], []
    reports_path = Path(reports_path)
    for report_file in reports_path.glob("*.json"):
        try:
            report = Report.load_from_base_report(report_file)
            if report.frames:
                target = [frame.meta["label"] for frame in report.frames]
                if sum(target) > 0:
                    reports.append(report)
                    targets.append(target)
        except JSONDecodeError:
            print(f"Reading report {report_file} failed")
            continue
    return reports, targets


def make_target(reports: List[Report]) -> List[List[int]]:
    targets = []
    for report in reports:
        target = [frame.meta["label"] for frame in report.frames]
        targets.append(target)
    return targets


def train(reports_path: str, save_path: str, model_name: Optional[str]):
    reports = []
    for file_name in iterate_reports(reports_path):
        report_path = os.path.join(reports_path, file_name)
        report = Report.load_report(report_path)
        if report.frames:
            if sum(frame.meta["label"] for frame in report.frames) > 1:
                reports.append(report)

    reports = reports

    target = make_target(reports)

    with open("config.json", "r") as f:
        config = json.load(f)

    model_names = ["scuffle", "deep_analyze", "code2seq"]

    if model_name:
        if model_name == "scuffle":
            encoder = ScuffleReportEncoder(**config["models"]["scuffle"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                **config["models"]["scuffle"]["tagger"]
            )
        elif model_name == "deep_analyze":
            encoder = TfIdfReportEncoder(**config["models"]["deep_analyze"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                **config["models"]["deep_analyze"]["tagger"]
            )
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
        # encoder = CachedReportEncoder("/home/dumtrii/Downloads/code2seq_embs")
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

    tagger = train_lstm_tagger(tagger, reports, target, **config["training"])

    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    train(args.reports_path, args.save_path, None)


if __name__ == '__main__':
    main()

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import List, Tuple, Optional

from new.data.report import Report
# from new.model.frame_encoders.code2seq import Code2SeqFrameEncoder
from new.model.lstm_tagger import LstmTagger
from new.model.report_encoders.cached_report_encoder import CachedReportEncoder
from new.model.report_encoders.scaffle_report_encoder import ScaffleReportEncoder
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


def train(reports_path: str, save_path: str, model_name: Optional[str]):
    reports, target = read_reports(reports_path)
    with open("config.json", "r") as f:
        config = json.load(f)

    if model_name:
        if model_name == "scaffle":
            encoder = ScaffleReportEncoder(**config["scaffle"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                **config["scaffle"]["tagger"]
            )
        elif model_name == "deep_analyze":
            encoder = TfIdfReportEncoder(**config["deep_analyze"]["encoder"]).fit(reports, target)
            tagger = LstmTagger(
                encoder,
                max_len=config["training"]["max_len"],
                **config["deep_analyze"]["tagger"]
            )
        else:
            raise ValueError("Wrong model type. Should be scaffle or deep_analyze")
    else:
        encoder = CachedReportEncoder("/home/dumtrii/Downloads/code2seq_embs")
        tagger = LstmTagger(
            encoder,
            max_len=config["training"]["max_len"],
            layers_num=2,
            hidden_dim=250
        )

    tagger = train_lstm_tagger(tagger, reports, target, **config["training"])

    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    train(args.reports_path, args.save_path, args.model)


if __name__ == '__main__':
    main()

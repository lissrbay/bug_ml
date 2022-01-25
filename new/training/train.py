import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import List, Tuple

from new.data.report import Report, Frame
# from new.model.frame_encoders.code2seq import Code2SeqFrameEncoder
from new.model.catboost_tagger import CatBoostTagger
from new.model.frame_encoders.tfidf import TfIdfFrameEncoder
from new.model.lstm_tagger import LstmTagger
from new.model.report_encoders.dummy_report_encoder import DummyReportEncoder
from new.model.report_encoders.simple_report_encoder import SimpleReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder
from new.training.torch_training import train_lstm_tagger
from new.model.report_encoders.scaffle_report_encoder import ScaffleReportEncoder


def read_reports(reports_path: str) -> Tuple[List[Report], List[List[int]]]:
    reports, targets = [], []
    reports_path = Path(reports_path)
    for report_file in reports_path.glob("*.json"):
        try:
            report = Report.load_from_base_report(report_file)
            reports.append(report)
            target = [frame.meta["label"] for frame in report.frames]
            targets.append(target)
        except JSONDecodeError:
            print(f"Reading report {report_file} failed")
            continue
    return reports, targets


def train(reports_path: str, save_path: str):
    reports, target = read_reports(reports_path)
    with open("config.json", "r") as f:
        config = json.load(f)

    # encoder = TfIdfReportEncoder(max_len=256).fit(reports, target)
    encoder = ScaffleReportEncoder(70, 100).fit(reports, target)

    tagger = LstmTagger(
        encoder,
        hidden_dim=250, layers_num=2, max_len=config["training"]["max_len"]
    )
    # tmp = tagger.predict(reports[0])
    tagger = train_lstm_tagger(tagger, reports, target, **config["training"])

    # tagger.fit(reports, target)
    #
    # cbst_tagger = CatBoostTagger(
    #     [tagger],
    #     DummyReportEncoder()
    # )

    # cbst_tagger.fit(reports, target)

    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    train(args.reports_path, args.save_path)


if __name__ == '__main__':
    main()

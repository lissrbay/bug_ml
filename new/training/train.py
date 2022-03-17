import argparse
import json
import sys
sys.path.insert(0, './../../')

from tqdm import tqdm
from typing import List, Tuple, Optional
from new.data.report import Report, Frame
# from new.model.frame_encoders.code2seq import Code2SeqFrameEncoder
from new.data_aggregation.utils import iterate_reports
from new.model.features.git_features import GitFeaturesTransformer
from new.model.features.metadata_features import MetadataFeaturesTransformer
from new.model.report_encoders.codebert_encoder import RobertaReportEncoder
from new.model.report_encoders.concat_encoders import ConcatReportEncoders
from new.model.lstm_tagger import LstmTagger
from new.model.report_encoders.cached_report_encoder import CachedReportEncoder
from new.model.report_encoders.scaffle_report_encoder import ScaffleReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder
from new.training.torch_training import train_lstm_tagger
from multiprocessing import Pool

import logging

loger = logging.getLogger('lightning')
loger.info(...)
loger.debug(...)


def read_report(report_path: str) -> Tuple[Report, List[int]]:
    report = Report.load_report(report_path)
    target = [frame.meta["label"] for frame in report.frames]

    return report, target


def read_reports(reports_path: str) -> Tuple[List[Report], List[List[int]]]:
    reports, targets = [], []
    # reports_path = Path(reports_path)
    pool = Pool()

    for report, target in pool.imap(read_report, tqdm(list(iterate_reports(reports_path)))):
        if sum(target) > 0:
            reports.append(report)
            targets.append(target)
    return reports, targets


def train(reports_path: str, save_path: str, model_name: Optional[str]):
    reports, target = read_reports(reports_path)
    with open("config.json", "r") as f:
        config = json.load(f)

    if model_name is not None:
        if model_name == "scuffle":
            encoder = ScaffleReportEncoder(**config["models"]["scuffle"]["encoder"]).fit(reports, target)
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
        else:
            raise ValueError("Wrong model type. Should be scaffle or deep_analyze")
    else:
        encoder = ConcatReportEncoders([RobertaReportEncoder(frames_count=config["training"]["max_len"], device='cuda'),
                                       #path_to_precomputed_embs="/home/lissrbay/Загрузки/code2seq_embs"),
                                       #GitFeaturesTransformer(
                                       #    frames_count=config["training"]["max_len"]).fit(reports, target),
                                       #MetadataFeaturesTransformer(frames_count=config["training"]["max_len"])
            ], device='cuda')
        tagger = LstmTagger(
            encoder,
            max_len=config["training"]["max_len"],
            layers_num=1,
            hidden_dim=120
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

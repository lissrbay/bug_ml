import argparse

from new.model.catboost_tagger import CatBoostTagger
from new.model.features.features_computer import FeaturesComputer
from new.model.frame_encoders.code2seq import Code2SeqFrameEncoder
from new.model.lstm_tagger import LstmTagger


def train(reports_path: str, save_path: str):
    tagger = CatBoostTagger(
        [LstmTagger(
            Code2SeqFrameEncoder("java"),
            hidden_dim=40, layers_num=1, with_crf=False, with_attention=False
        )],
        FeaturesComputer(["code", "one_hot_exception"])
    )
    reports, target = read_reports(reports_path)
    tagger.fit(reports, target)
    tagger.save(save_path)
    return tagger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    train(args.reports_path, args.save_path)


if __name__ == '__main__':
    main()

import argparse
from typing import List

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger


class BugLocApi:
    _name: str
    _tagger: BlamedTagger

    @classmethod
    def download(cls, name: str):
        pass

    @classmethod
    def initialize(cls, name: str):
        name = name or BugLocApi._name
        BugLocApi._tagger = BlamedTagger.load(name)

    @classmethod
    def predict(cls, report: Report) -> List[float]:
        return BugLocApi._tagger.predict(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=False, default=None)
    args = parser.parse_args()

    BugLocApi.download(args.load_path)
    BugLocApi.initialize(args.load_path)


if __name__ == '__main__':
    main()

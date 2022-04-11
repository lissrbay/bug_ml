from abc import ABC, abstractmethod
from typing import List, Union

from new.data.report import Report
from new.model.blamed_ranker import BlamedRanker
from new.model.blamed_tagger import BlamedTagger
from new.model.report_encoders.report_encoder import ReportEncoder


class DummyRanker(BlamedRanker):
    def __init__(self, feature_sources: List[Union[BlamedTagger, ReportEncoder]]):
        self.feature_sources = feature_sources

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'BlamedRanker':
        return self

    @abstractmethod
    def predict(self, report: Report) -> List[float]:
        for feature_source in self.feature_sources:
            if isinstance(feature_source, BlamedTagger):
                return feature_source.predict(report)

        raise ValueError('No blamed tagger!')

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'BlamedRanker':
        pass

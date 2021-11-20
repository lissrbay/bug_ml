from abc import ABC, abstractmethod
from typing import List

from new.data.report import Report


class BlamedTagger(ABC):
    def fit(self, reports: List[Report], target: List[List[int]]) -> 'BlamedTagger':
        return self

    @abstractmethod
    def predict(self, report: Report) -> List[float]:
        raise NotImplementedError

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> 'BlamedTagger':
        pass

from typing import List

from new.data.report import Report
from new.model.features.feature import BaseFeature


class CodeFeatures(BaseFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CodeFeatures':
        return self

    def compute(self, report: Report) -> List[List[float]]:
        pass

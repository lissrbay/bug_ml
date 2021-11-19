from typing import List

from new.data.report import Report
from new.model.features.feature import BaseFeature


class CodeFeatures(BaseFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, report: Report) -> List[List[float]]:
        pass

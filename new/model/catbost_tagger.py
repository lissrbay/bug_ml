from typing import List

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger


class CatBoostTagger(BlamedTagger):
    """
    Combine predictions of other models, add features and get final CatBoost-based model
    """

    def __init__(self, models: List[BlamedTagger], feature_computer):
        self.models = models
        self.feature_computer = feature_computer

    def predict(self, report: Report) -> List[float]:
        pass

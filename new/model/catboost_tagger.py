from typing import List

from catboost import CatBoost

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.features.features_computer import FeaturesComputer
from new.model.report_encoders.report_encoder import ReportEncoder


class CatBoostTagger(BlamedTagger):
    """
    Combine predictions of other models, add features and get final CatBoost-based model
    """

    default_params = {
        'iterations': 200,
        'depth': 5,
        'loss_function': 'QuerySoftMax',
        'custom_metric': ['PrecisionAt:top=2', 'RecallAt:top=2', 'MAP:top=2'],
        'eval_metric': 'AverageGain:top=2',  # eval_metric='AUC',
        'metric_period': 100
    }

    def __init__(self, models: List[BlamedTagger], report_encoder: ReportEncoder):
        self.taggers = models
        self.report_encoder = report_encoder
        self.model = CatBoost(CatBoostTagger.default_params)

    def create_features(self, report: Report) -> List[List[float]]:
        features = self.report_encoder.encode_report(report).tolist()
        for tagger in self.taggers:
            frame_preds = tagger.predict(report)
            for i, value in enumerate(frame_preds[:len(features)]):
                features[i].append(value)
        return features

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CatBoostTagger':
        self.report_encoder.fit(reports, target)
        for tagger in self.taggers:
            tagger.fit(reports, target)
        features = [self.create_features(report) for report in reports]
        self.model.fit(features, target)
        return self

    def predict(self, report: Report) -> List[float]:
        features = self.create_features(report)
        return self.model.predict(features)

    def save(self, path: str):
        self.model.save_model(path, format="onnx")

    @classmethod
    def load(cls, path) -> 'CatBoostTagger':
        pass

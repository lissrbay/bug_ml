from typing import List, Union

import numpy as np
from catboost import CatBoost

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.features.features_computer import FeaturesComputer
from new.model.report_encoders.report_encoder import ReportEncoder


class CatBoostRanker(BlamedRanker):
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

    def __init__(self, feature_sources: List[Union[BlamedTagger, ReportEncoder]]):
        self.feature_sources = feature_sources

    def create_features(self, report: Report) -> List[List[float]]:
        features = self.report_encoder.encode_report(report).tolist()
        for tagger in self.taggers:
            frame_preds = tagger.predict(report)
            for i, value in enumerate(frame_preds[:len(features)]):
                features[i].append(value)
        return features

    def train_test_splitting(self, features, targets, grouping, fraction=0.9):
        reports_count = len(list(set(grouping)))
        train_count = int(reports_count * fraction)

        last_group = -1

        for i in grouping:
            if i != last_group:

        train_reports, test_reports = report_ids[:train_count], report_ids[train_count:]
        df_val = df_features[df_features['report_id'].isin(test_reports)].drop(
            ['method_name', 'indices', 'exception_class'], axis=1)

        df_features = df_features[df_features['report_id'].isin(train_reports)]
        X, test_X, y, test_y = train_test_split(
            df_features.drop(['label', 'method_name', 'indices', 'exception_class'], axis=1), df_features['label'],
            test_size=0.1, shuffle=False)
        train_dataset = Pool(X.drop(["report_id"], axis=1), y, group_id=X['report_id'])
        test_dataset = Pool(test_X.drop(["report_id"], axis=1), test_y, group_id=test_X['report_id'])
        return train_dataset, test_dataset, df_val

    def get_features(self, reports: List[Report]):
        features = []

        for feature_source in self.feature_sources:
            source_features = []
            for report in reports:
                if isinstance(feature_source, BlamedTagger):
                    source_features.append(feature_source.predict(report).reshape(-1, 1))
                elif isinstance(feature_source, ReportEncoder):
                    source_features.append(feature_source.encode_report(report))
            features.append(source_features)

        report_ids_for_grouping = []
        for report in reports:
            for _ in report.frames:
                report_ids_for_grouping.append(report.id)

        return np.vstack(features), report_ids_for_grouping

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CatBoostTagger':
        features, grouping = self.get_features(reports)

        return self

    def predict(self, report: Report) -> List[float]:
        features = self.create_features(report)
        return self.model.predict(features)

    def save(self, path: str):
        self.model.save_model(path, format="onnx")

    @classmethod
    def load(cls, path) -> 'CatBoostTagger':
        pass

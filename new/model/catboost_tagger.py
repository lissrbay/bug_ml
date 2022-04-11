import pickle
from typing import List, Union

import numpy as np
from catboost import CatBoost, Pool

from model.blamed_tagger import BlamedTagger
from new.data.report import Report
from new.model.blamed_ranker import BlamedRanker
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

    def __init__(self, feature_sources: List[Union[BlamedTagger, ReportEncoder]],
                save_path: str):
        self.feature_sources = feature_sources
        self.save_path = save_path

    def create_features(self, report: Report) -> List[List[float]]:
        features = self.report_encoder.encode_report(report).tolist()
        for tagger in self.taggers:
            frame_preds = tagger.predict(report)
            for i, value in enumerate(frame_preds[:len(features)]):
                features[i].append(value)
        return features

    def split_by_report_id(self, grouping, train_count):
        last_group = -1
        border = (0, 0)
        for i, report_id in enumerate(grouping):
            if report_id != last_group:
                last_group = report_id
                border = (border[0] + 1, border[1])

            if border[0] == train_count:
                break
            border = (border[0], border[1] + 1)

        return border[1]

    def train_test_splitting(self, features, targets, grouping, fraction=0.9):
        reports_count = len(list(set(grouping)))
        train_count = int(reports_count * fraction)

        border = self.split_by_report_id(grouping, train_count)

        train_features, test_features = features[:border], features[border:]
        train_targets, test_targets = targets[:border], targets[border:]
        train_groups, test_groups = grouping[:border], grouping[border:]

        train_dataset = Pool(train_features, train_targets, group_id=train_groups)
        test_dataset = Pool(test_features, test_targets, group_id=test_groups)

        return train_dataset, test_dataset

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

    def fit(self, reports: List[Report], targets: List[List[int]]) -> 'CatBoostTagger':
        features, grouping = self.get_features(reports)
        test_pool, train_pool = self.train_test_splitting(features, targets, grouping)

        self.model = CatBoost(self.default_params)
        self.model.fit(train_pool, eval_set=test_pool)
        self.save(self.save_path)
        return self

    def predict(self, report: Report) -> List[float]:
        features = self.create_features(report)
        return self.model.predict(features)

    def save(self, path: str):
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path) -> 'CatBoostTagger':
        return pickle.load(open(path, 'rb'))

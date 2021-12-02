from typing import List, Dict, Callable

from new.data.report import Report
from new.model.features.code_features import CodeFeatures
from new.model.features.feature import BaseFeature
from new.model.features.one_hot_exception import OneHotExceptionClass


class FeaturesComputer:
    features: Dict[str, Callable[[Dict], BaseFeature]] = {
        "code": lambda kwargs: CodeFeatures(**kwargs),
        "one_hot_exception": lambda kwargs: OneHotExceptionClass(**kwargs)
    }

    def __init__(self, feature_names: List[str], **kwargs):
        self.feature_names = feature_names
        self.features: Dict[str, BaseFeature] = {
            name: FeaturesComputer.features[name](**kwargs) for name in self.feature_names
        }

    def fit(self, reports: List[Report], target: List[List[int]]):
        for name, feature in self.features:
            feature.fit(reports, target)

    def compute(self, report: Report) -> List[List[float]]:
        feature_value = []
        for name in self.feature_names:
            feature_value += self.features[name].compute(report)
        return feature_value

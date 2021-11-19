from typing import List, Dict, Callable

from new.data.report import Report
from new.model.features.code_features import CodeFeatures
from new.model.features.feature import BaseFeature


class FeaturesComputer:
    features: Dict[str, Callable[[Dict], BaseFeature]] = {
        "code": lambda kwargs: CodeFeatures(**kwargs)
    }

    def __init__(self, feature_names: List[str], **kwargs):
        self.feature_names = feature_names
        self.features: Dict[str, BaseFeature] = {
            name: FeaturesComputer.features[name](**kwargs) for name in self.feature_names
        }

    def compute(self, report: Report) -> List[List[float]]:
        feature_value = []
        for name in self.feature_names:
            feature_value += self.features[name].compute(report)
        return feature_value

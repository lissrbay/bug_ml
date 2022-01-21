from typing import List

from new.data.report import Report, Frame
from new.model.features.feature import BaseFeature
import numpy as np


class MetadataFeaturesTransformer(BaseFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exception_class = []
        self.has_runs = []
        self.has_dollars = []
        self.is_parallel = []
        self.method_file_position = []
        self.is_java_standart = []
        self.label = []
        self.method_stack_position = []

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CodeFeatures':
        pass

    def extract_method_name_features(self, method_name: str):
        self.has_runs.append("run" in method_name)
        self.has_dollars.append("$" in method_name)
        self.is_parallel.append("Thread" in method_name)
        self.is_java_standart.append(method_name[:4] == 'java')

    def extract_exception_class(self, report: Report):
        self.exception_class.append(report.exceptions)

    def extract_method_file_position(self, frame: Frame):
        self.method_file_position.append(int(frame.meta['line_number']))

    def transform(self, report: Report) -> List[List[float]]:
        for i, frame in enumerate(report.frames):
            method_name = frame.meta['method_name']
            self.extract_method_name_features(method_name)

            self.extract_exception_class(report)

            self.extract_method_file_position(frame)

            self.method_stack_position.append(i)

            self.label.append(frame.meta['label'])

        return list(np.vstack(self.exception_class,
                              self.has_runs,
                              self.has_dollars,
                              self.is_parallel,
                              self.method_file_position,
                              self.is_java_standart,
                              self.label,
                              self.method_stack_position,
                              ))

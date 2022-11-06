from typing import List

import torch
from torch import Tensor

from new.data.report import Report, Frame
from new.model.report_encoders.report_encoder import ReportEncoder


class MetadataFeaturesTransformer(ReportEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.feature_names = [ 
            'has_runs',
            'has_dollars',
            'is_parallel',
            'method_file_position',
            'is_java_standard']

    def extract_method_name_features(self, frames: List[Frame], method_name_features):
        for frame in frames:
            method_name = frame.meta['method_name']
            method_name_features['has_runs'].append("run" in method_name)
            method_name_features['has_dollars'].append("$" in method_name)
            method_name_features['is_parallel'].append("Thread" in method_name)
            method_name_features['is_java_standard'].append(method_name[:4] == 'java')
        return method_name_features

    def extract_exception_class(self, report: Report, features):
        return features['exceptions'].extend([report.exceptions for _ in report.frames[:self.frames_count]])

    def extract_method_position(self, frames: List[Frame], features):
        return features['method_stack_position'].extend([i for i in range(len(frames))])

    def extract_method_file_position(self, frames: List[Frame], features):
        for frame in frames:
            if 'line' in frame.meta:
                features['method_file_position'].append(0 if frame.meta['line'] is None else int(frame.meta['line']))
            else:
                raise Exception('No field line in frame.meta')
        return features

    def encode_report(self, report: Report) -> Tensor:
        features = {k: [] for k in self.feature_names}
        frames = report.frames
        features = self.extract_method_name_features(frames, features)
        features = self.extract_method_file_position(frames, features)
        report_features = [torch.FloatTensor(features[name]) for name in self.feature_names]
        report_features = torch.cat(report_features, dim=0).T
        return report_features.reshape(len(report.frames), len(features)).to(self.device)

    @property
    def dim(self) -> int:
        return len(self.feature_names)

from typing import List, Dict, Callable

import torch
from torch import Tensor, nn

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder
from new.data_aggregation.utils import pad_features

class ConcatReportEncoders(ReportEncoder, nn.Module):
    def __init__(self, report_encoders: List[ReportEncoder], **kwargs):
        super().__init__()

        assert len(report_encoders) > 0
        self.report_encoders = nn.ModuleList(report_encoders)
        self.device = kwargs['device']
        self.frames_count = kwargs['frames_count']

    def fit(self, reports: List[Report], target: List[List[int]]):
        for name, feature in self.features.items():
            feature.fit(reports, target)

    def encode_report(self, report: Report) -> Tensor:
        feature_value = []
        for encoder in self.report_encoders:
            feature_value += [pad_features(encoder.encode_report(report).to(self.device), self.frames_count)]

        return torch.cat(feature_value, dim=1)

    @property
    def dim(self) -> int:
        encoders_dims = [report_encoder.dim for report_encoder in self.report_encoders]
        assert len(encoders_dims) > 0
        return sum(encoders_dims)


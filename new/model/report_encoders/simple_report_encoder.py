from typing import List

import torch
from torch import Tensor

from new.data.report import Report
from new.model.frame_encoders.frame_encoder import FrameEncoder
from new.model.report_encoders.report_encoder import ReportEncoder


class SimpleReportEncoder(ReportEncoder):
    def __init__(self, frame_encoder: FrameEncoder):
        self.frame_encoder = frame_encoder

    def encode_report(self, report: Report) -> Tensor:
        return torch.vstack([self.frame_encoder.encode(frame) for frame in report.frames])

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        self.frame_encoder.fit(reports, target)
        return self

    @property
    def dim(self) -> int:
        return self.frame_encoder.dim()

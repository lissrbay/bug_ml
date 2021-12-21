import torch
from torch import Tensor

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class DummyReportEncoder(ReportEncoder):
    def __init__(self, dim: int = 320):
        self._dim = dim

    def encode_report(self, report: Report) -> Tensor:
        return torch.zeros(len(report.frames), self.dim)

    @property
    def dim(self) -> int:
        return self._dim

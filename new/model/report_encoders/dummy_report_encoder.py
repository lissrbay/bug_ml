from typing import List

from torch import Tensor

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class DummyReportEncoder(ReportEncoder):
    def encode_report(self, report: Report) -> Tensor:
        pass

    @property
    def dim(self) -> int:
        return 320

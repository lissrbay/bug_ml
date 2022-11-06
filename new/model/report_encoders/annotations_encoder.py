from typing import List

import numpy as np
import torch
from torch import Tensor

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class AnnotationsEncoder(ReportEncoder):
    def __init__(self, caching: bool = False, **kwargs):
        super().__init__()

        self.caching = caching
        self.device = kwargs.get('device', 'cpu')

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        return self

    def encode_report(self, report: Report) -> Tensor:
        """ Returns [seq_len; feature_size] tensor. """
        report_times = []
        report_max_time = report.exception_time
        for frame in report.frames:
            method_max_time = np.max(frame.meta['ts'])
            has_code = frame.code.code != ''
            if has_code and report_max_time > 0:
                frame_time = np.log(report_max_time - method_max_time + 1)
                if np.isnan(frame_time):
                    frame_time = 0.0
            else:
                frame_time = 0.0

            report_times.append(frame_time)

        report_times = np.array(report_times)
        report_times = torch.FloatTensor(report_times)

        result = report_times.reshape(report_times.shape[0], 1).to(self.device)

        return result

    @property
    def dim(self) -> int:
        return 1

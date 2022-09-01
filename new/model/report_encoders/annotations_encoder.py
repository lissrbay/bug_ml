from typing import List

import numpy as np
import torch
from torch import Tensor

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class AnnotationsEncoder(ReportEncoder):
    def __init__(self, caching: bool = False, **kwargs):
        self.caching = caching
        self.device = kwargs.get('device', 'cpu')
        self.report_cache = {}

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        return self

    def encode_report(self, report: Report) -> Tensor:
        """ Returns [seq_len; feature_size] tensor. """
        report_times = []
        for frame in report.frames:
            report_max_time = frame.meta['report_max_time']

            method_max_time = frame.meta['method_time_max'] / 1000
            has_code = frame.code.code != ''
            if has_code and report_max_time > 0:
                frame_time = method_max_time - report_max_time
                if np.isnan(frame_time):
                    frame_time = 0.0
            else:
                frame_time = 0.0

            report_times.append(frame_time)

        report_times = np.array(report_times)
        report_times = report_times / np.max(report_times) if np.max(report_times) > 0 else report_times
        report_times = torch.FloatTensor(report_times)

        result = report_times.reshape(report_times.shape[0], 1).to(self.device)
        if self.caching:
            self.report_cache[report.id] = result
            return self.report_cache[report.id]
        else:
            return result

    @property
    def dim(self) -> int:
        return 1

from typing import List

import numpy as np
import torch
from torch import Tensor

from new.data.report import Report, Frame
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
        self._max_value = 120
        self._to_month = lambda x: x / 1_000 / 60 / 60 / 24 / 30

        result = []
        for frame in report.frames:
            if frame.authored_date is None:
                result.append(self._max_value)

            max_timestamp = int(frame.authored_date)

            if max_timestamp <= 0:
                result.append(self._max_value)

            elapsed_time_ms = report.timestamp - max_timestamp
            if elapsed_time_ms <= 0:
                return [self._max_value]

            return [self._to_month(elapsed_time_ms)]


        report_times = []
        report_max_time = max(frame.authored_date for frame in report.frames)
        for frame in report.frames:
            # report_max_time = frame.meta['report_max_time']

            method_max_time = frame.meta['method_time_max'] / 1000
            method_max_time = frame.authored_date
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


class JBReportFeatures(ReportEncoder):
    def __init__(self):
        super(JBReportFeatures, self).__init__(2)
        self._jb_names = ["jetbrains", "intellij"]

    def _featurize_frame(self, frame: Frame) -> List[float]:
        frame_name = frame.method.lower()
        return [float(jb_name in frame_name) for jb_name in self._jb_names]

    def encode_report(self, report: Report, **kwargs) -> List[List[float]]:
        return [self._featurize_frame(frame) for frame in report.frames]


class PositionReportFeatures(ReportEncoder):
    def __init__(self):
        super(PositionReportFeatures, self).__init__(2)

    def _featurize_frame(self, position: int, stack_len: int) -> List[float]:
        return [position, position / stack_len]

    def __call__(self, report: Report, **kwargs) -> List[List[float]]:
        return [self._featurize_frame(i, len(report)) for i in range(len(report))]


class TimestampReportFeatures(Report):
    def __init__(self):
        super(TimestampReportFeatures, self).__init__(1)
        self._max_value = 120
        self._to_month = lambda x: x / 1_000 / 60 / 60 / 24 / 30

    def _featurize_frame(self, frame: Frame, report_ts: int) -> List[float]:
        annotation = frame.annotation

        if annotation is None:
            return [self._max_value]

        max_timestamp = int(np.max(annotation.timestamps))

        if max_timestamp <= 0:
            return [self._max_value]

        elapsed_time_ms = report_ts - max_timestamp
        if elapsed_time_ms <= 0:
            return [self._max_value]

        return [self._to_month(elapsed_time_ms)]

    def __call__(self, report: Report, **kwargs) -> List[List[float]]:
        return [self._featurize_frame(frame, report.timestamp) for frame in report.frames]


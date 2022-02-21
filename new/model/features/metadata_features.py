from typing import List
import torch
from torch.nn.functional import pad
from torch import FloatTensor, Tensor

from new.data.report import Report, Frame
from new.model.report_encoders.report_encoder import ReportEncoder
import numpy as np


class MetadataFeaturesTransformer(ReportEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.exception_class = []
        self.has_runs = []
        self.has_dollars = []
        self.is_parallel = []
        self.method_file_position = []
        self.is_java_standart = []
        self.label = []
        self.method_stack_position = []
        self.frames_count = kwargs['frames_count']

    def extract_method_name_features(self, method_name: str):
        self.has_runs.append("run" in method_name)
        self.has_dollars.append("$" in method_name)
        self.is_parallel.append("Thread" in method_name)
        self.is_java_standart.append(method_name[:4] == 'java')

    def extract_exception_class(self, report: Report):
        self.exception_class.append(report.exceptions)

    def extract_method_file_position(self, frame: Frame):
        if 'line' in frame.meta:
            self.method_file_position.append(0 if frame.meta['line'] is None else int(frame.meta['line']))
        else:
            raise Exception('No field line in frame.meta')

    def encode_report(self, report: Report) -> Tensor:
        self.exception_class = []
        self.has_runs = []
        self.has_dollars = []
        self.is_parallel = []
        self.method_file_position = []
        self.is_java_standart = []
        self.label = []
        self.method_stack_position = []

        for i, frame in enumerate(report.frames[:self.frames_count]):
            method_name = frame.meta['method_name']
            self.extract_method_name_features(method_name)

            self.extract_exception_class(report)

            self.extract_method_file_position(frame)

            self.method_stack_position.append(i)
        pad_size = self.frames_count - min(len(report.frames), self.frames_count)
        return pad(FloatTensor(np.vstack(#self.exception_class,
                              [self.has_runs,
                              self.has_dollars,
                              self.is_parallel,
                              self.method_file_position,
                              self.is_java_standart,]
                              #self.method_stack_position,
                              ).T), (0, 0, 0, pad_size))

    @property
    def dim(self) -> int:
        return 5

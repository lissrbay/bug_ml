from abc import ABC, abstractmethod
from typing import List

from torch import Tensor

from new.data.report import Report


class ReportEncoder(ABC):
    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        return self

    @abstractmethod
    def encode_report(self, report: Report) -> Tensor:
        """ Returns [seq_len; feature_size] tensor. """
        raise NotImplementedError

    def encode_static(self, report: Report) -> Tensor:
        """ Data preprocessing for dataset. """
        return self.encode_report(report)

    def encode_trainable(self, inputs: Tensor, mask: Tensor) -> Tensor:
        """ Trainable part of encoding. """
        return inputs

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

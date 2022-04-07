from abc import ABC, abstractmethod
from typing import List

from torch import Tensor, nn

from new.data.report import Report

class ReportEncoder(ABC, nn.Module):
    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        return self

    @abstractmethod
    def encode_report(self, report: Report) -> Tensor:
        """ Returns [seq_len; feature_size] tensor. """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import List
from torch import Tensor
from new.data.report import Report


class ReportEncoder(ABC):
    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        return self

    @abstractmethod
    def encode_report(self, report: Report) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

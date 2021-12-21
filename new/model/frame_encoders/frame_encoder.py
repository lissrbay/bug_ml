from abc import ABC, abstractmethod
from typing import List

import torch

from new.data.report import Frame, Report


class FrameEncoder(ABC):
    def fit(self, reports: List[Report], target: List[List[int]]) -> 'FrameEncoder':
        return self

    @abstractmethod
    def encode(self, frame: Frame) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class ConcatFrameEncoder(FrameEncoder):
    def __init__(self, frame_encoders: List[FrameEncoder]):
        self.frame_encoders = frame_encoders
        self._dim = sum(encoder.dim for encoder in self.frame_encoders)

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ConcatFrameEncoder':
        for encoder in self.frame_encoders:
            encoder.fit(reports, target)
        return self

    def encode(self, frame: Frame) -> torch.Tensor:
        pass

    @property
    def dim(self) -> int:
        return self._dim

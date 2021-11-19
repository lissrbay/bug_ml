import base64
from abc import ABC, abstractmethod
from typing import List

import torch

from new.data.report import Frame


class FrameEncoder(ABC):
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

    def encode(self, frame: Frame) -> torch.Tensor:
        pass

    def dim(self) -> int:
        return self._dim

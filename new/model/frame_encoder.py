from abc import ABC, abstractmethod
from typing import List

from new.data.report import Frame


class FrameEncoder(ABC):
    @abstractmethod
    def encode(self, frame: Frame):
        raise NotImplementedError


class ConcatFrameEncoder(FrameEncoder):
    def __init__(self, frame_encoders: List[FrameEncoder]):
        self.frame_encoders = frame_encoders

    def encode(self, frame: Frame):
        pass


class TfIdfFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass


class Code2SeqFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass


class ScaffleFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass

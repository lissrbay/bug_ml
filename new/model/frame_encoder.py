from abc import ABC, abstractmethod

from new.data.report import Frame


class FrameEncoder(ABC):
    @abstractmethod
    def encode(self, frame: Frame):
        raise NotImplementedError


class TfIdfFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass


class Code2SeqFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass


class ScaffleFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame):
        pass

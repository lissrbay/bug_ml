import torch

from new.data.report import Frame
from new.model.frame_encoders.frame_encoder import FrameEncoder


class TfIdfFrameEncoder(FrameEncoder):
    def encode(self, frame: Frame) -> torch.Tensor:
        pass

    def dim(self) -> int:
        pass

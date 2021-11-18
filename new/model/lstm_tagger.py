from typing import List

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.frame_encoder import FrameEncoder


class LstmTagger(BlamedTagger):
    def __init__(self, frame_encoder: FrameEncoder, with_crf: bool = True, with_attention: bool = True):
        self.frame_encoder = frame_encoder
        self.with_crf = with_crf
        self.with_attention = with_attention

    def predict(self, report: Report) -> List[float]:
        pass

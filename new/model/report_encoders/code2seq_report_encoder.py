import base64
import os
from functools import lru_cache

import numpy as np
import torch
from pycode2seq.inference.model.model import Model as EmbModel

from new.data.report import Report, Frame
from new.model.report_encoders.report_encoder import ReportEncoder
from new.data_aggregation.pycode2seq_embeddings import embed_frames
from new.constants import EMBEDDING_DIM, EMBEDDING_TMP_FILE


class Code2SeqFrameEncoder(ReportEncoder):
    _tmp_file_name = EMBEDDING_TMP_FILE
    _emb_dim = EMBEDDING_DIM

    def __init__(self, name: str, frames_limit=80):
        self.emb_model = EmbModel.load(name)
        self.frames_limit = frames_limit

    @lru_cache(maxsize=10000)
    def encode_report(self, report: Report) -> torch.Tensor:
        embeddings = embed_frames(self.emb_model, report)
        return embeddings

    def dim(self) -> int:
        return Code2SeqFrameEncoder._emb_dim

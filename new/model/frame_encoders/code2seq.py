import base64
import os
from functools import lru_cache

import numpy as np
import torch
from pycode2seq.inference.model.model import Model as EmbModel

from new.data.report import Frame
from new.model.frame_encoders.frame_encoder import FrameEncoder
from new.constants import EMBEDDING_DIM, EMBEDDING_TMP_FILE


class Code2SeqFrameEncoder(FrameEncoder):
    _tmp_file_name = EMBEDDING_TMP_FILE
    _emb_dim = EMBEDDING_DIM

    def __init__(self, name: str):
        self.emb_model = EmbModel.load(name)

    @lru_cache(maxsize=10000)
    def encode(self, frame: Frame) -> torch.Tensor:
        code = frame.get_code_decoded()
        if code != "":
            with open(Code2SeqFrameEncoder._tmp_file_name, "w") as f:
                f.write(code)
            # TODO: return full method name
            method_embeddings = self.emb_model.methods_embeddings(Code2SeqFrameEncoder._tmp_file_name)
            os.remove(Code2SeqFrameEncoder._tmp_file_name)
            method_short_name = frame.name.split('.')[-1]
            if method_short_name in method_embeddings:
                embedding = method_embeddings[method_short_name].detach().numpy()
            else:
                embedding = np.zeros(Code2SeqFrameEncoder._emb_dim)
        else:
            embedding = np.zeros(Code2SeqFrameEncoder._emb_dim)
        return torch.FloatTensor(embedding)

    def dim(self) -> int:
        return Code2SeqFrameEncoder._emb_dim

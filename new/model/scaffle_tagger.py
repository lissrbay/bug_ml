from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.frame_encoders.frame_encoder import FrameEncoder


class ScaffleTagger(BlamedTagger):
    def __init__(self, frame_encoder: FrameEncoder, hidden_dim: int, layers_num: int = 1,
                 with_crf: bool = False, with_attention: bool = False, num_classes=0):
        self.frame_encoder = frame_encoder
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.with_crf = with_crf
        self.with_attention = with_attention
        self.num_classes = num_classes

        self.rnn_line_level = nn.RNN(frame_encoder.dim, self.hidden_dim, batch_first=True,
                                     num_layers=self.layers_num, bidirectional=True)
        self.rnn_trace_level = nn.RNN(self.hidden_dim, 125,
                                      num_layers=self.layers_num, bidirectional=True)
        self.tagger = nn.Linear(250, 1)

    def forward(self, inputs):
        n = inputs.shape[0]
        max_methods = inputs.shape[2]
        res, hn = self.rnn_line_level(inputs.reshape(-1, max_methods, self.frame_encoder.dim()))
        res, _ = self.rnn_trace_level(hn[0].reshape(n, -1, self.lstm_hidden_dim))

        tag = self.tagger(res)
        tag = tag.reshape(-1, self.num_classes)
        return torch.sigmoid(tag).reshape(-1, self.num_classes)

    def predict(self, report: Report) -> List[float]:
        frames_embeddings = []
        for frame in report.frames:
            frames_embeddings.append(self.frame_encoder.encode(frame))
        frames_embeddings = torch.cat(frames_embeddings, axis=0)
        pred = self.forward(frames_embeddings)
        return pred.reshape(self.num_classes)

    @classmethod
    def load(cls, path: str) -> 'ScaffleTagger':
        pass

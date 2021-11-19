from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.frame_encoders.frame_encoder import FrameEncoder


class ExtendedEmbedding(nn.Embedding):
    def __init__(self, vocab_size=2, word_emb_dim=384):
        super().__init__(vocab_size, word_emb_dim)
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim

    def embeddings(self, inputs):
        emb = self(torch.LongTensor([[0]]))
        inputs[np.where(~inputs.detach().numpy().all(axis=2))] = emb
        emb = self(torch.LongTensor([[1]]))
        inputs[np.where(np.sum(inputs.detach().numpy(), axis=2) == self.word_emb_dim)] = emb

        return inputs


class LstmTagger(BlamedTagger):
    def __init__(self, frame_encoder: FrameEncoder, hidden_dim: int, layers_num: int = 1,
                 with_crf: bool = False, with_attention: bool = False):
        self.frame_encoder = frame_encoder
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.with_crf = with_crf
        self.with_attention = with_attention

        self.word_emb = ExtendedEmbedding(2, self.frame_encoder.dim)
        self.lstm = nn.LSTM(self.frame_encoder.dim, self.hidden_dim,
                            num_layers=self.layers_num, bidirectional=True)
        self.tagger = nn.Linear(2 * self.hidden_dim, 2)

    def forward(self, inputs):
        embeddings = self.word_emb.embeddings(inputs)
        res, _ = self.lstm(embeddings)
        tag = self.tagger(res)
        return F.softmax(tag, 1)

    def predict(self, report: Report) -> List[float]:
        pass

    @classmethod
    def load(cls, path: str) -> 'LstmTagger':
        pass

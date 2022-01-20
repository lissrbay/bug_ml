import logging
from typing import List

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import Tensor, nn
from torch.nn.functional import pad

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def clean_method_name(method_name):
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


class ScaffleReportEncoder(ReportEncoder, nn.Module):
    def __init__(self, dim: int, word2vec_dim: int, device: str = "cuda"):
        super().__init__()
        self._dim = dim
        self.word2vec_dim = word2vec_dim
        self.device = device

        self.word2vec = Word2Vec(vector_size=word2vec_dim, workers=11, min_count=1000)
        self.lstm = nn.LSTM(self.word2vec_dim, self._dim, num_layers=1, bidirectional=True)

    def encode_report(self, report: Report) -> Tensor:
        encoded_tokens = []
        for frame in report.frames:
            method_name_tokens = TfIdfReportEncoder.tokenize(clean_method_name(frame.name))
            embeddings = []
            for word in method_name_tokens:
                try:
                    vector = self.word2vec.wv[word].reshape((self.word2vec_dim,))
                except KeyError:
                    vector = np.zeros((self.word2vec_dim,))
                vector = torch.tensor(vector, device=self.device, dtype=torch.float32)
                embeddings.append(vector)

            encoded_tokens.append(torch.vstack(embeddings))

        lengths = [enc.shape[0] for enc in encoded_tokens]
        max_len = max(lengths)
        encoded_tokens = [pad(enc, (0, 0, 0, max_len - enc.shape[0])).unsqueeze(1) for enc in encoded_tokens]
        encoding, _ = self.lstm(torch.cat(encoded_tokens, dim=1))

        return encoding[torch.tensor(lengths, device=self.device) - 1, torch.arange(len(report.frames))]

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        sentences = [list(TfIdfReportEncoder.tokenize(clean_method_name(frame.name)))
                     for report in reports for frame in report.frames]
        self.word2vec.build_vocab(sentences, progress_per=1000)
        self.word2vec.train(sentences, total_examples=len(sentences), epochs=30, report_delay=1)
        return self

    @property
    def dim(self) -> int:
        return self._dim * 2

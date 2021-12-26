import numpy as np
import torch
from gensim.models import Word2Vec
from torch import Tensor, nn

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.tfidf import TfIdfReportEncoder


def clean_method_name(method_name):
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


class ScaffleReportEncoder(ReportEncoder, nn.Module):
    def __init__(self, dim: int, word2vec_dim: int):
        super().__init__()
        self._dim = dim
        self.word2vec_dim = word2vec_dim
        self.word2vec = Word2Vec(vector_size=word2vec_dim)
        self.lstm = nn.LSTM(self.word2vec_dim, self._dim, num_layers=1, bidirectional=True)

    def encode_report(self, report: Report) -> Tensor:
        encoding = []
        for frame in report.frames:
            method_name_tokens = TfIdfReportEncoder.tokenize(clean_method_name(frame.name))
            embeddings = []
            for word in method_name_tokens:
                try:
                    vector = self.emb_model.wv[word].reshape((100,))
                except KeyError:
                    vector = np.zeros((100,))
                embeddings.append(vector)
            # embedding = self.
            encoding.append(torch.FloatTensor(embeddings))

    @property
    def dim(self) -> int:
        return self._dim * 2
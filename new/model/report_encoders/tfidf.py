from re import split
from typing import List

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.utils import tokenize_frame


class TfIdfReportEncoder(ReportEncoder):
    def __init__(self, max_len: int, min_df=0.1) -> None:
        super().__init__()
        self.min_df = min_df
        self.max_len = max_len

    @staticmethod
    def _split_reports(reports: List[Report]):
        method_docs = []
        namespace_docs = []
        for report in reports:
            tokens = [frame.name.split(".") for frame in report.frames]
            method_docs.append(".".join(frame_tokens[-1] for frame_tokens in tokens))
            namespace_docs.append(".".join(".".join(frame_tokens[:-1]) for frame_tokens in tokens))

        return method_docs, namespace_docs

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'ReportEncoder':
        self.method_vectorizer = TfidfVectorizer(tokenizer=tokenize_frame, min_df=self.min_df)
        self.namespace_vectorizer = TfidfVectorizer(tokenizer=tokenize_frame, min_df=self.min_df)

        method_docs, namespace_docs = self._split_reports(reports)

        self.method_vectorizer = self.method_vectorizer.fit(method_docs)
        self.namespace_vectorizer = self.namespace_vectorizer.fit(namespace_docs)

        return self

    def encode_report(self, report: Report) -> Tensor:
        tokens = [frame.name.split(".") for frame in report.frames[:self.max_len]]

        method_embeddings = self.method_vectorizer.transform([frame_tokens[-1] for frame_tokens in tokens]).todense()
        namespace_emdeddings = self.namespace_vectorizer.transform(
            [".".join(frame_tokens[:-1]) for frame_tokens in tokens]).todense()

        return torch.cat((torch.Tensor(method_embeddings), torch.Tensor(namespace_emdeddings)), dim=-1)

    @property
    def dim(self) -> int:
        return len(self.method_vectorizer.vocabulary_) + len(self.namespace_vectorizer.vocabulary_)

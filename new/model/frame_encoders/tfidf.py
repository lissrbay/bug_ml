from re import split
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from new.data.report import Frame, Report
from new.model.frame_encoders.frame_encoder import FrameEncoder


class TfIdfFrameEncoder(FrameEncoder):
    def __init__(self, min_df=0.1) -> None:
        super().__init__()
        self.min_df = min_df

    @staticmethod
    def split_into_subtokens(name: str):
        return [word.lower() for word in split(r'(?=[A-Z])', name) if word]

    @staticmethod
    def tokenize(doc: str):
        return (word.lower() for token in doc.split(".") for word in TfIdfFrameEncoder.split_into_subtokens(token))
    
    @staticmethod
    def _split_reports(reports: List[Report]):
        method_docs = []
        namespace_docs = []
        for report in reports:
            tokens = [frame.name.split(".") for frame in report.frames]
            method_docs.append(".".join(frame_tokens[-1] for frame_tokens in tokens))
            namespace_docs.append(".".join(".".join(frame_tokens[:-1]) for frame_tokens in tokens))

        return method_docs, namespace_docs

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'FrameEncoder':
        self.method_vectorizer = TfidfVectorizer(tokenizer=TfIdfFrameEncoder.tokenize, min_df=self.min_df)
        self.namespace_vectorizer = TfidfVectorizer(tokenizer=TfIdfFrameEncoder.tokenize, min_df=self.min_df)
        
        method_docs, namespace_docs = self._split_reports(reports)

        self.method_vectorizer = self.method_vectorizer.fit(method_docs)
        self.namespace_vectorizer = self.namespace_vectorizer.fit(namespace_docs)
        
    def encode(self, frame: Frame) -> torch.Tensor:
        tokens = [frame.name.split(".")]

        method_embeddings = self.method_vectorizer.transform([frame_tokens[-1] for frame_tokens in tokens]).todense()
        namespace_emdeddings = self.namespace_vectorizer.transform([".".join(frame_tokens[:-1]) for frame_tokens in tokens]).todense()

        return torch.cat((torch.Tensor(method_embeddings), torch.Tensor(namespace_emdeddings)), dim=-1)

    @property
    def dim(self) -> int:
        return len(self.method_vectorizer.vocabulary_) + len(self.namespace_vectorizer.vocabulary_)

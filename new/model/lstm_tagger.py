from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import pad
from torchcrf import CRF

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.modules.deep_analyze import DeepAnalyzeAttention
from new.model.report_encoders.report_encoder import ReportEncoder


def get_label_scores(crf: CRF, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n, batch_size, _ = emissions.size()

    scores = torch.zeros(emissions.shape, dtype=emissions.dtype, device=emissions.device)

    for j in range(n):
        scores[j] = crf.transitions[labels[j - 1], :].clone()

    scores = scores + emissions
    # scores = nn.Softmax(dim=-1)(scores)

    return scores * mask.float().unsqueeze(-1)


class LstmTagger(BlamedTagger, nn.Module):
    def __init__(self, report_encoder: ReportEncoder, hidden_dim: int, max_len: int, layers_num: int = 1,
                 with_crf: bool = False, with_attention: bool = False, scaffle: bool = False, device: str = "cuda"):
        super().__init__()
        self.report_encoder = report_encoder
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.with_crf = with_crf
        self.with_attention = with_attention
        self.scaffle = scaffle
        self.max_len = max_len
        self.device = device

        self.lstm = nn.LSTM(self.report_encoder.dim, self.hidden_dim,
                            num_layers=self.layers_num, bidirectional=True)
        self.tagger = nn.Linear(2 * self.hidden_dim, 2)

        if with_crf:
            self.crf = CRF(2)

        if with_attention:
            self.attention = DeepAnalyzeAttention(2 * self.hidden_dim, 2, max_len)
            self.lstm_dropout = nn.Dropout(0.25)

        if scaffle:
            self.scaffle_final = nn.Linear(2 * self.hidden_dim, max_len)

    def calc_emissions(self, report: Report, mask: torch.Tensor) -> torch.Tensor:
        features = self.report_encoder.encode_report(report).to(self.device)
        features = features[:self.max_len]
        features = pad(features, (0, 0, 0, self.max_len - features.shape[0])).unsqueeze(1)
        embeddings = features * mask.unsqueeze(-1)
        res, _ = self.lstm(embeddings)

        if self.scaffle:
            return torch.sigmoid(self.scaffle_final(res[-1, :, :]).permute(1, 0))

        if self.with_attention:
            # res = self.lstm_dropout(res)
            res = self.attention(res, mask)
        else:
            res = self.tagger(res)

        return res

    def forward(self, reports: List[Report], masks: Optional[List[torch.Tensor]]):
        if masks is None:
            emissions = [self.forward_single(report) for report in reports]
        else:
            emissions = [self.forward_single(report, mask) for report, mask in zip(reports, masks)]
        return torch.cat(emissions, dim=1)

    def forward_single(self, report: Report, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.tensor([True] * self.max_len).unsqueeze(-1).to(self.device)
        mask = mask[:self.max_len]
        seq_len, batch_size = mask.shape
        emissions = self.calc_emissions(report, mask)

        if self.with_crf:
            preds = self.crf.decode(emissions, mask)

            preds = [pred + [0] * (seq_len - len(pred)) for pred in preds]
            preds = torch.tensor(preds).transpose(0, 1).to(emissions.device)

            return get_label_scores(self.crf, emissions, preds, mask)

        return emissions

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'BlamedTagger':
        return self

    def predict(self, report: Report) -> List[float]:
        with torch.no_grad():
            preds = self.forward_single(report)
            preds = preds[:, 0, 1]

        return preds.tolist()


    @classmethod
    def load(cls, path: str) -> 'LstmTagger':
        pass


if __name__ == "__main__":
    pass
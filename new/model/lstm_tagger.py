from pathlib import Path
from typing import List, Any

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch import nn
from torch.optim import Adam
from torchmetrics import MetricCollection

from new.data.report import Report
from new.model.blamed_tagger import BlamedTagger
from new.model.frame_encoders.frame_encoder import FrameEncoder

from torchcrf import CRF

from new.model.modules.deep_analyze import DeepAnalyzeAttention

from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.training import ReportsDataModule, Precision, Recall, TopkAccuracy
import pytorch_lightning as pl


def get_label_scores(crf: CRF, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    n, batch_size, _ = emissions.size()

    scores = torch.zeros(emissions.shape, dtype=emissions.dtype, device=emissions.device)

    for j in range(n):
        scores[j] = crf.transitions[labels[j-1], :].clone()

    scores = scores + emissions
    # scores = nn.Softmax(dim=-1)(scores)

    return scores * mask.float().unsqueeze(-1)

class ExtendedEmbedding(nn.Embedding):
    def __init__(self, vocab_size=2, word_emb_dim=384):
        super().__init__(vocab_size, word_emb_dim)
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim

    def embeddings(self, inputs):
        emb = self(torch.LongTensor([[0]]).to(inputs.device))
        inputs[np.where(~inputs.cpu().detach().numpy().all(axis=2))] = emb
        emb = self(torch.LongTensor([[1]]).to(inputs.device))
        inputs[np.where(np.sum(inputs.cpu().detach().numpy(), axis=2)
                        == self.word_emb_dim)] = emb

        return inputs


class LstmTagger(BlamedTagger, pl.LightningModule):
    def __init__(self, report_encoder: ReportEncoder, hidden_dim: int, max_len: int, layers_num: int = 1,
                 with_crf: bool = False, with_attention: bool = False):
        super().__init__()
        self.report_encoder = report_encoder
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.with_crf = with_crf
        self.with_attention = with_attention
        self.max_len = max_len

        self.word_emb = ExtendedEmbedding(2, self.report_encoder.dim)
        self.lstm = nn.LSTM(self.report_encoder.dim, self.hidden_dim,
                            num_layers=self.layers_num, bidirectional=True)
        self.tagger = nn.Linear(2 * self.hidden_dim, 2)

        if with_crf:
            self.crf = CRF(2)

        if with_attention:
            self.attention = DeepAnalyzeAttention(2 * self.hidden_dim, 2, max_len)
            self.lstm_dropout = nn.Dropout(0.25)

    def calc_emissions(self, inputs, mask):
        # embeddings = self.word_emb.embeddings(inputs)
        embeddings = inputs * mask.unsqueeze(-1)
        res, _ = self.lstm(embeddings)

        if self.with_attention:
            res = self.lstm_dropout(res)
            res = self.attention(res, mask)
        else:
            res = self.tagger(res)

        return res

    def forward(self, inputs, mask):
        seq_len, batch_size = mask.shape
        emissions = self.calc_emissions(inputs, mask)


        if self.with_crf:
            preds = self.crf.decode(emissions, mask)

            preds = [pred + [0] * (seq_len - len(pred)) for pred in preds]
            preds = torch.tensor(preds).transpose(0, 1).to(inputs.device)
            
            return get_label_scores(self.crf, emissions, preds, mask)

        return emissions


    def fit(self, reports: List[Report], target: List[List[int]]) -> 'BlamedTagger':
        reports_path = Path("/home/dumtrii/Downloads/bug_ml_computed")
        reports_path = None
        datamodule = ReportsDataModule(reports, target, 4, self.report_encoder, 80, reports_path)
        model = TrainingModule(self)

        trainer = Trainer(gpus=1)
        trainer.fit(model, datamodule)

        return self

    def predict(self, report: Report) -> List[float]:
        pass

    @classmethod
    def load(cls, path: str) -> 'LstmTagger':
        pass

class TrainingModule(pl.LightningModule):
    def __init__(self, tagger: LstmTagger):
        super().__init__()
        self.tagger = tagger

        self.train_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(3)])
        self.val_metrics = MetricCollection([Precision(), Recall(), TopkAccuracy(3)])

    def training_step(self, batch, batch_idx):
        inputs, labels, mask = batch

        if self.tagger.with_crf:
            emissions = self.tagger.calc_emissions(inputs, mask)
            loss = -self.tagger.crf(emissions, labels, mask)
        else:
            scores = self.tagger.forward(inputs, mask)
            loss = nn.functional.cross_entropy(scores.transpose(1, 2), labels, ignore_index=2)

        with torch.no_grad():
            scores = self.tagger.forward(inputs, mask)
            preds = scores.argmax(dim=-1)

        self.train_metrics.update(preds, labels, mask, scores=scores)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args):
        inputs, labels, mask = batch

        if self.tagger.with_crf:
            emissions = self.tagger.calc_emissions(inputs, mask)
            loss = -self.tagger.crf(emissions, labels, mask)
        else:
            scores = self.tagger.forward(inputs, mask)
            loss = nn.functional.cross_entropy(scores.transpose(1, 2), labels, ignore_index=2)

        with torch.no_grad():
            scores = self.tagger.forward(inputs, mask)
            preds = scores.argmax(dim=-1)

        self.val_metrics.update(preds, labels, mask, scores=scores)

        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        super().validation_epoch_end(outputs)
        self.log("val_metrics", self.val_metrics.compute())
        self.val_metrics.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        super().training_epoch_end(outputs)
        self.log("train_metrics", self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    # from sys
    emb = ExtendedEmbedding()
    x = [[[0, 1, 0, 0, 1]]]
    print(emb.embeddings(torch.LongTensor(x)))
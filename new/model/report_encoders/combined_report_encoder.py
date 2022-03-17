import torch
import torch.nn as nn

from new.data.report import Report
from new.model.report_encoders.code2seq_report_encoder import Code2SeqReportEncoder
from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.scuffle_report_encoder import ScuffleReportEncoder


class CombinedReportEncoder(ReportEncoder, nn.Module):
    def __init__(self, code2seq_encoder: Code2SeqReportEncoder, scuffle_encoder: ScuffleReportEncoder):
        super().__init__()
        self.code2seq_encoder = code2seq_encoder
        self.scuffle_encoder = scuffle_encoder

    def encode_report(self, report: Report) -> torch.Tensor:
        code2seq_embeddings = self.code2seq_encoder.encode_report(report)
        scuffle_embeddings = self.scuffle_encoder.encode_report(report)
        return torch.cat((code2seq_embeddings, scuffle_embeddings), dim=1)

    @property
    def dim(self) -> int:
        return self.code2seq_encoder.dim + self.scuffle_encoder.dim

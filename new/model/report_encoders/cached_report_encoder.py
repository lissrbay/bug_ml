from new.data.report import Report

from torch import Tensor
from torch.nn.functional import pad
import torch

from new.model.report_encoders.report_encoder import ReportEncoder


class CachedReportEncoder(ReportEncoder):
    def __init__(self, path_to_precomputed_embs: str):
        super().__init__()
        self.precomputed_embs = torch.load(path_to_precomputed_embs)

    def encode_report(self, report: Report) -> Tensor:
        report_id = report.id
        if report_id in self.precomputed_embs:
            report_embs = self.precomputed_embs[report_id]
            pad_size = self.dim() - len(report_embs)
            return pad(report_embs, (pad_size, self.emb_dim()))

        return self.torch.empty((self.dim(), self.emb_dim()))

    def dim(self):
        return self.precomputed_embs.shape[1]

    def emb_dim(self):
        return self.precomputed_embs.shape[-1]

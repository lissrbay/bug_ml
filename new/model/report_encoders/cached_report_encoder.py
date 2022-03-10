from new.data.report import Report

from torch import Tensor
from torch.nn.functional import pad
import torch

from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.extended_embedding import ExtendedEmbedding


class CachedReportEncoder(ReportEncoder):
    def __init__(self, path_to_precomputed_embs: str, use_pad_embs: bool = False, **kwargs):
        super().__init__()
        self.precomputed_embs = torch.load(path_to_precomputed_embs)
        self.word_emb = None
        if use_pad_embs:
            self.word_emb = ExtendedEmbedding(2, self.dim)

    def encode_report(self, report: Report) -> Tensor:
        report_id = str(report.id)
        if report_id in self.precomputed_embs:
            report_embs = self.precomputed_embs[report_id]
            pad_size = self.frame_count - len(report_embs)
            report_embs = pad(report_embs, (0, pad_size, 0, 0))
        else:
            report_embs = torch.zeros((self.frame_count, self.dim))
        if self.word_emb is not None:
            report_embs = self.word_emb.embeddings(report_embs)

        return report_embs

    @property
    def frame_count(self):
        return self.precomputed_embs.frames_count

    @property
    def dim(self):
        return self.precomputed_embs.dim

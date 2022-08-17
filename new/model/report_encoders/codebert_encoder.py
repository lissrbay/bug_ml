import torch
from torch import Tensor
from transformers import RobertaTokenizer, RobertaModel

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class RobertaReportEncoder(ReportEncoder, torch.nn.Module):
    BERT_MODEL_DIM = 768

    def __init__(self, frames_count: int, caching: bool = False, **kwargs):
        super().__init__()
        self.frames_count = frames_count
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = kwargs.get('device', 'cpu')
        self.model.to(self.device)
        self.report_cache = {}
        self.caching = caching

    def encode_report(self, report: Report) -> Tensor:
        report_embs = []
        if report.id in self.report_cache and self.caching:
            return self.report_cache[report.id]
        for frame in report.frames:
            method_code = frame.get_code_decoded()
            if method_code:
                if not self.caching:
                    code_tokens = self.tokenizer.tokenize(method_code[:512])
                    tokens_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(code_tokens))[None, :]
                    context_embeddings = self.model(tokens_ids.to(self.device), )[0]
                    vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
                    del tokens_ids, code_tokens
                else:
                    with torch.no_grad():
                        code_tokens = self.tokenizer.tokenize(method_code[:512])
                        tokens_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(code_tokens))[None, :]
                        context_embeddings = self.model(tokens_ids.to(self.device))[0]
                        vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
                        del tokens_ids, code_tokens
            else:
                vec = torch.zeros((self.BERT_MODEL_DIM,)).to(self.device).requires_grad_()
            report_embs.append(vec)
        self.report_cache[report.id] = torch.cat(report_embs).reshape(-1, self.BERT_MODEL_DIM)
        return self.report_cache[report.id]

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM

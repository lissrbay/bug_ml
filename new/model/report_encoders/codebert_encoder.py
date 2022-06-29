import re

import torch
from torch import Tensor
from transformers import RobertaTokenizer, RobertaModel

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class RobertaReportEncoder(ReportEncoder):
    BERT_MODEL_DIM = 768

    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.model.to(self.device)



    def encode_report(self, report: Report) -> Tensor:
        report_embs = []
        for frame in report.frames:
            code = frame.get_code_decoded()
            method_code = self.extract_method_code(code, clean_method_name(frame.meta['method_name']))
            if method_code:
                with torch.no_grad():
                    code_tokens = self.tokenizer.tokenize(method_code[:512])
                    tokens_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(code_tokens))[None, :]
                    context_embeddings = self.model(tokens_ids.to(self.device))[0]
                    vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
                    del tokens_ids, code_tokens
            else:
                vec = torch.zeros((self.BERT_MODEL_DIM,)).to(self.device)
            report_embs.append(vec)
        return torch.cat(report_embs).reshape(-1, self.BERT_MODEL_DIM)

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM

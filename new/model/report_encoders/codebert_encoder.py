import numpy as np
import torch
from torch import Tensor
from transformers import RobertaTokenizer, RobertaModel
from random import random
from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class RobertaReportEncoder(ReportEncoder):
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
        report_times = []

        for frame in report.frames:
            method_code = frame.get_code_decoded()
            report_max_time = frame.meta['report_max_time']

            method_max_time = frame.meta['method_time_max']/1000
            has_code = frame.code.code != ''
            #print(method_max_time)
            if has_code and report_max_time > 0:
                frame_time = method_max_time-report_max_time
                if np.isnan(frame_time):
                    frame_time = 0.0
            else:
                frame_time = 0.0
            #print(frame_time, type(frame_time))
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
            #vec = torch.FloatTensor(list(vec) + [frame_time + 0.0001])
            report_embs.append(vec)
            report_times.append(frame_time)
        report_times = np.array(report_times)
        report_times = report_times / np.max(report_times) if np.max(report_times) > 0 else report_times
        report_times = torch.FloatTensor(report_times).to(self.device)

        report_embs = torch.cat(report_embs).reshape(-1, self.BERT_MODEL_DIM)
        self.report_cache[report.id] = torch.cat([report_embs, report_times.reshape(report_embs.shape[0], 1)], axis=1)
        return self.report_cache[report.id]

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM + 1

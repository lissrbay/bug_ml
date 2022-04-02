from importlib.metadata import requires
import re

import torch
from torch import Tensor, nn
from transformers import RobertaTokenizer, RobertaModel

import numpy as np

from new.data.report import Report
from new.data_aggregation.parser_java_kotlin import Parser
from new.model.report_encoders.report_encoder import ReportEncoder


def remove_tabs(code):
    # code = list(filter(lambda x: not (x.strip()[:2] == '//'), code))
    # code = '\n'.join(code)
    code = re.sub(' +', ' ', code)
    return re.sub('\t+', '', code)


def code_fragment(bounds, code):
    if not bounds:
        return ''
    if bounds[1] <= bounds[0]:
        return ''
    return ''.join(code)[bounds[0]: bounds[1]]


def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


class RobertaReportEncoder(ReportEncoder, nn.Module):
    BERT_MODEL_DIM = 768

    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.mode = kwargs['mode']
        self.model.to(self.device)

    def extract_method_code(self, code, method_name):
        parser = Parser()
        txt = remove_tabs(code)
        ast = parser.parse(txt)
        method_info = ast.get_method_names_and_bounds()
        code = ''
        for name, bounds in method_info:
            name_ = name.split(':')[-1]
            if method_name in name:
                method_code = code_fragment(bounds[0], txt)
                code = name_ + method_code

        return code

    def random_frames_for_autograd(self, report, k=25):
        take_rand = np.zeros(len(report.frames))

        if self.mode == 'train':
            frame_ids = []
            label_pos = -1
            for i, frame in enumerate(report.frames):
                if frame.get_code_decoded():
                    frame_ids.append(i)
                    if label_pos == -1 and frame.meta["label"] == 1:
                        label_pos = i
            take_rand[np.random.choice(frame_ids, min(len(frame_ids), k), replace=False)] = 1
            if label_pos != -1:
                take_rand[label_pos] = 1

        return take_rand

    def encode_report(self, report: Report) -> Tensor:
        report_embs = []
        take_rand = self.random_frames_for_autograd(report)
        for i, frame in enumerate(report.frames[:self.frames_count]):
            code = frame.get_code_decoded()
            method_code = self.extract_method_code(code, clean_method_name(frame.meta['method_name']))
            if method_code:
                code_tokens = self.tokenizer.tokenize(method_code[:512])
                tokens_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(code_tokens))[None, :]
                context_embeddings = self.model(tokens_ids.to(self.device))[0]
                vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
                del tokens_ids, code_tokens
            else:
                vec = torch.zeros((self.BERT_MODEL_DIM,))
            vec =  torch.tensor(vec, requires_grad=bool(take_rand[i])).to(self.device)
            report_embs.append(vec)

        return torch.cat(report_embs).reshape(-1, self.BERT_MODEL_DIM)

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM

from new.data.report import Report

from torch import Tensor
from torch.nn.functional import pad
import torch

from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.extended_embedding import ExtendedEmbedding
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import os
from tqdm import tqdm
import json
from new.data_aggregation.parser_java_kotlin import Parser
import re

def load_report(name):
    f = open(name, 'r')
    report = json.load(f)
    f.close()
    return report

def remove_tabs(code):
    #code = list(filter(lambda x: not (x.strip()[:2] == '//'), code))
    #code = '\n'.join(code)
    code = re.sub(' +', ' ', code)
    return re.sub('\t+', '', code)

def code_fragment( bounds, code):
    if not bounds:
        return ''
    if bounds[1]<= bounds[0]:
        return ''
    return ''.join(code)[bounds[0]: bounds[1]]

def get_method_info(self, ast):
    methods_info = ast.get_method_names_and_bounds()
    methods_info = dict(methods_info)
    return methods_info

def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name

class RobertaReportEncoder(ReportEncoder):
    BERT_MODEL_DIM = 768
    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.model.to(self.device)

    def extract_method_code(self, code, method_name):
        parser = Parser()
        txt = remove_tabs(code)
        ast = parser.parse(txt)
        method_info = ast.get_method_names_and_bounds()
        #print(method_info)
        code = ''
        for name, bounds in method_info:
            name_ = name.split(':')[-1]
            #print(method_name, name)
            if method_name in name:
                method_code = code_fragment(bounds[0], txt)
                code = name_ + method_code

        return code

    def encode_report(self, report: Report) -> Tensor:
        report_embs = []
        for frame in report.frames:
            code = frame.get_code_decoded()
            method_code = self.extract_method_code(code, clean_method_name(frame.meta['method_name']))
            if method_code:
                with torch.no_grad():
                    code_tokens = self.tokenizer.tokenize(method_code[:512])
                    tokens_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(code_tokens))[None,:]
                    context_embeddings = self.model(tokens_ids.to(self.device))[0]
                    vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
                    del tokens_ids, code_tokens
            else:
                vec = torch.zeros((self.BERT_MODEL_DIM,)).to(self.device)
            report_embs.append(vec)
        return pad(torch.cat(report_embs).reshape(-1, self.BERT_MODEL_DIM), (0, 0, 0, 80 - len(report.frames)))

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM

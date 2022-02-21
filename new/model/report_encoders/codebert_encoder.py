from new.data.report import Report

from torch import Tensor
from torch.nn.functional import pad
import torch

from new.model.report_encoders.report_encoder import ReportEncoder
from new.model.report_encoders.extended_embedding import ExtendedEmbedding
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = "cuda"


class RobertaReportEncoder(ReportEncoder):
    BERT_MODEL_DIM = 768
    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.model.to(device)

    def encode_report(self, report: Report) -> Tensor:
        report_embs = []
        for frame in report.frames:
            code = frame.get_code_decoded()
            code_tokens = self.tokenizer.tokenize(code.replace("\n", '')[:256])
            tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
            context_embeddings = self.model(torch.tensor(tokens_ids)[None,:].to(device))[0]
            vec = context_embeddings.mean(axis=1).reshape((self.BERT_MODEL_DIM,))
            del tokens_ids, code_tokens
            report_embs.append(vec)
        return torch.FloatTensor(torch.cat(report_embs))

    @property
    def frame_count(self):
        return self.frames_count

    @property
    def dim(self):
        return self.BERT_MODEL_DIM

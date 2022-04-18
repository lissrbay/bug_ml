import torch

from torch import nn
import numpy as np


class ExtendedEmbedding(nn.Embedding):
    def __init__(self, vocab_size=2, word_emb_dim=384, device='cpu'):
        super().__init__(vocab_size, word_emb_dim)
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        self.device = device

    def embeddings(self, inputs):
        emb1 = self(torch.LongTensor([[0]]).to(self.device))
        inputs[torch.where((inputs.detach() == 0).all(dim=1))] = emb1
        emb2 = self(torch.LongTensor([[1]]).to(self.device))
        inputs[torch.where(inputs.detach().sum(dim=1) == self.word_emb_dim)] = emb2
        return inputs

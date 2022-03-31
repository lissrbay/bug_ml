import torch

from torch import nn
import numpy as np


class ExtendedEmbedding(nn.Embedding):
    def __init__(self, vocab_size=2, word_emb_dim=384):
        super().__init__(vocab_size, word_emb_dim)
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim

    def embeddings(self, inputs):
        emb = self(torch.LongTensor([[0]]).to(inputs.device))
        inputs[np.where((inputs.cpu().detach().numpy() == 0).all(axis=1))] = emb
        emb = self(torch.LongTensor([[1]]).to(inputs.device))
        inputs[np.where(np.sum(inputs.cpu().detach().numpy(), axis=1)
                        == self.word_emb_dim)] = emb
        return inputs

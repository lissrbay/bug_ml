import torch.nn as nn
import torch
import numpy as np


class ExtendedEmbedding(nn.Embedding):
    def __init__(self, vocab_size = 2, word_emb_dim = 384):
        super().__init__(vocab_size, word_emb_dim)
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim
        
        
    def embeddings(self, inputs):
        emb = self(torch.LongTensor([[0]]))
        inputs[np.where(~inputs.detach().numpy().all(axis=2))] = emb
        emb = self(torch.LongTensor([[1]]))
        inputs[np.where(np.sum(inputs.detach().numpy(), axis=2)==self.word_emb_dim)] = emb
        return inputs
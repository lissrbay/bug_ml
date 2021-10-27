import torch.nn as nn
import torch.nn.functional as F
from .extended_embeddings_layer import ExtendedEmbedding


class LSTMTagger(nn.Module):
    def __init__(self, word_emb_dim=384, lstm_hidden_dim=40, lstm_layers_count=1):
        super().__init__()
        self.word_emb_dim = word_emb_dim

        self.word_emb = ExtendedEmbedding(2, word_emb_dim)
        self.lstm = nn.LSTM(word_emb_dim, lstm_hidden_dim,
                            num_layers=lstm_layers_count, bidirectional=True)
        self.tagger = nn.Linear(2*lstm_hidden_dim, 2)

    def forward(self, inputs):
        embeddings = self.word_emb.embeddings(inputs)
        res, _ = self.lstm(embeddings)
        tag = self.tagger(res)
        return F.softmax(tag, 1)
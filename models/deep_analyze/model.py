from unicodedata import bidirectional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from modules import DeepAnalyzeAttention, DeepAnalyzeCRF

def accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    correct = torch.sum(pred[mask] == labels[mask], dim=0)
    return correct.sum() / mask.sum()

class DeepAnalyze(pl.LightningModule):
    def __init__(self, feature_size, lstm_hidden_size, lstm_num_layers, n_tags):
        super().__init__()
        self.padding = 0
        self.bi_listm = nn.LSTM(feature_size, lstm_hidden_size, 
                                    num_layers=lstm_num_layers, bidirectional=True)
        self.attention = DeepAnalyzeAttention(lstm_hidden_size * 2, n_tags)
        self.crf = DeepAnalyzeCRF(n_tags)

    def forward(self, inputs, mask):
        x, _ = self.bi_listm(inputs)
        x = self.attention(x, mask)
        x = self.crf(x, mask)
        return x

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        x, _ = self.bi_listm(x)
        x = self.attention(x, mask)
        x = self.crf.neg_log_likelihood(x, y, mask)
        return x.mean()

    # def validation_step(self, batch, *args):
    #     x, y, mask = batch
    #     feats, _ = self.bi_listm(x)
    #     att = self.attention(feats, mask)
    #     nll = self.crf.neg_log_likelihood(att, y, mask)
    #     with torch.no_grad():
    #         _, preds, _ = self.forward(x, mask)

    #     self.log("Neg log likelihood", nll.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
    #     self.log("Accuracy", accuracy(preds, y, mask), on_step=True, on_epoch=False, prog_bar=True, logger=True)
    #     return nll.mean(), 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


# if __name__ == "__main__":

#     da = DeepAnalyze(5, 3, 1, 2)

#     crf = DeepAnalyzeCRF(2)
#     lengths = torch.tensor([1, 2, 2, 3])
#     a = torch.randn((3, 4, 2))
#     b = torch.randint(0, 2, (3, 4))
#     mask = torch.arange(3).unsqueeze(-1).expand(3, len(lengths)) < lengths.long().unsqueeze(0)


#     loss = crf.neg_log_likelihood(a, b).mean()
#     print(loss)
#     loss.backward()
    
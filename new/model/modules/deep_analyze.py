import torch
import torch.nn as nn
import pytorch_lightning as pl


class DeepAnalyzeAttention(pl.LightningModule):
    def __init__(self, input_dim, output_dim, max_len):
        super().__init__()
        self.prob_layer = nn.Linear(input_dim, output_dim)
        self.weight = nn.parameter.Parameter(torch.randn(output_dim, max_len))
        self.g_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.result_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        # feats: [max_len; batch_size; input_dim]

        # [max_len; batch_size; output_dim]
        probs = self.prob_layer(feats)

        probs = probs * mask.unsqueeze(-1)

        # [output_dim; batch_size; output_dim]
        scores = torch.einsum("xh,hby->xby", self.weight, probs)
        a = self.softmax(scores)

        # [max_len; batch_size; outout_dim]
        h = torch.einsum("xby,nby->nbx", a, probs)

        # [seq_len; batch_size; input_dim]
        # combined = torch.cat((g, feats), dim=-1)
        # z = self.g_linear(combined)

        h = h * mask.unsqueeze(-1)

        # [seq_len; batch_size; output_dim]
        return h
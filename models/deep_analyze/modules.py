from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
from re import split

from sklearn.feature_extraction.text import TfidfVectorizer


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec: torch.Tensor):
    max_score, _ = vec.max(dim=1)
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score.unsqueeze(-1))))

class DeepAnalyzeAttention(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.score_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.g_linear = nn.Linear(input_dim * 2, input_dim, bias=False)
        self.result_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        # inputs: [seq_len; batch_size; input_dim]

        feats = feats * mask.unsqueeze(-1)

        # [seq_len; seq_len; batch_size; input_dim]
        scores = torch.einsum("xbh,ybh->xybh", feats, feats)
        scores = self.score_linear(scores)
        a = self.softmax(scores)

        # [seq_len; batch_size; input_dim]
        g = torch.einsum("xybh,ybh->xbh", a, feats)

        # [seq_len; batch_size; input_dim]
        combined = torch.cat((g, feats), dim=-1)
        z = self.g_linear(combined)

        z = z * mask.unsqueeze(-1)

        # [seq_len; batch_size; output_dim]
        return self.result_linear(z)

class DeepAnalyzeCRF(pl.LightningModule):
    def __init__(self, n_tags):
        super().__init__()
        self.n_special_tags = 3
        self.n_tags = n_tags + self.n_special_tags
        self.padding = 0
        self.start_tag_id = 1
        self.stop_tag_id = 2
        self.ninf = -100000

        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))

        self.transitions.data[self.start_tag_id, :] = self.ninf
        self.transitions.data[:, self.stop_tag_id] = self.ninf
        # self.transitions.data[self.padding, :] = self.ninf
        self.transitions.data[:, self.padding] = self.ninf

        self.softmax = nn.Softmax(dim=-1)


    def _forward_alg(self, feats: torch.Tensor, mask: torch.Tensor):
        # feats: [seq_len; batch_size; n_tags]
        seq_len, batch_size, _ = feats.shape
        
        lengths = mask.sum(dim=0)

        # Add special tags
        special_tags = torch.zeros((seq_len, batch_size, self.n_special_tags)).to(feats.device)
        special_tags[0, :, self.start_tag_id] = 1
        special_tags[lengths - 1, :, self.stop_tag_id] = 1
        special_tags[mask][self.padding] = 1

        feats = torch.cat((special_tags, feats), dim=-1)
        
        # Start on start tag
        init_alphas = torch.full((batch_size, self.n_tags), self.ninf).to(feats.device)
        init_alphas[:, self.start_tag_id] = 0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        terminal_var = torch.zeros((batch_size, self.n_tags)).to(forward_var.device)

        for i, feat in enumerate(feats):
            alphas_t = []
            for next_tag in range(self.n_tags):
                # Scores from attention layer
                emit_score = feat[:, next_tag].unsqueeze(-1)

                # Transition scores
                trans_score = self.transitions[next_tag]
                

                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(-1))


            forward_var = torch.cat(alphas_t, dim=-1).view(batch_size, -1)
            terminal_var[i == lengths] = forward_var[i == lengths]
        
        terminal_var[lengths == seq_len] = forward_var[lengths == seq_len]
        terminal_var = terminal_var + self.transitions[self.stop_tag_id]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, mask):
        # Gives the score of a provided tag sequence
        seq_len, batch_size = tags.shape

        lengths = mask.sum(dim=0)

        tags = tags + self.n_special_tags

        special_tags = torch.zeros((seq_len, batch_size, self.n_special_tags)).to(feats.device)
        special_tags[0, :, self.start_tag_id] = 1
        special_tags[lengths-1, :, self.stop_tag_id] = 1
        special_tags[mask][self.padding] = 1

        feats = torch.cat((special_tags, feats), dim=-1)

        score = torch.zeros(batch_size).to(feats.device)
        tags = torch.cat([torch.full((1, batch_size), self.start_tag_id, dtype=torch.long).to(feats.device), tags])
        for i, feat in enumerate(feats):
            step_scores = self.transitions[tags[i + 1], tags[i]] + feat[torch.arange(batch_size), tags[i + 1]]
            step_scores = step_scores * mask[i]
            score = score + \
                step_scores
        step_scores = self.transitions[self.stop_tag_id, tags[-1]]
        step_scores = step_scores * mask[i]
        score = score + step_scores
        return score

    def _viterbi_decode(self, feats, mask):
        backpointers = []
        # feats: [seq_len; batch_size; n_tags]
        seq_len, batch_size, _ = feats.shape

        lengths = mask.sum(dim=0)

        # Add special tags
        special_tags = torch.zeros((seq_len, batch_size, self.n_special_tags)).to(feats.device)
        special_tags[0, :, self.start_tag_id] = 1
        special_tags[lengths - 1, :, self.stop_tag_id] = 1
        special_tags[mask][self.padding] = 1

        feats = torch.cat((special_tags, feats), dim=-1)

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((batch_size, self.n_tags), self.ninf).to(feats.device)
        init_vvars[:, self.start_tag_id] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        terminal_var = torch.zeros((batch_size, self.n_tags)).to(forward_var.device)
        for i, feat in enumerate(feats):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.n_tags):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var, dim=-1).reshape(-1)
                bptrs_t.append(best_tag_id.reshape(-1, 1))
                viterbivars_t.append(next_tag_var[torch.arange(batch_size), best_tag_id].reshape(-1, 1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t, dim=-1) + feat).view(batch_size, -1)
            bptrs_t = torch.cat(bptrs_t, dim=-1)
            bptrs_t[lengths < i] = self.padding
            terminal_var[i == lengths] = forward_var[i == lengths]

            backpointers.append(bptrs_t)

        
        # Transition to STOP_TAG
        terminal_var[seq_len == lengths] = forward_var[seq_len == lengths]
        terminal_var = terminal_var + self.transitions[self.stop_tag_id]
        terminal_best_tag_id = torch.argmax(terminal_var, dim=-1).reshape(-1)

        best_tag_id = terminal_best_tag_id.clone().to(terminal_best_tag_id.device)
        best_tag_id[lengths < seq_len] = self.padding

        path_score = terminal_var[torch.arange(batch_size), terminal_best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        backpointers = torch.cat([bptr.unsqueeze(0) for bptr in backpointers])

        for i, bptrs_t in enumerate(reversed(backpointers)):
            best_tag_id[lengths == seq_len - i] = terminal_best_tag_id[lengths == seq_len - i]
            best_tag_id = bptrs_t[torch.arange(batch_size), best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert (start == self.start_tag_id).all()  # Sanity check
        best_path.reverse()

        return path_score, torch.vstack(best_path) - self.n_special_tags    

    def neg_log_likelihood(self, feats, tags, mask):
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)

        return forward_score, gold_score

    def forward(self, feats, mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats, mask)
        return score, tag_seq, mask


class TfidfEmbeddings:
    def __init__(self, min_df=0.01) -> None:
        self.method_vectorizer = TfidfVectorizer(tokenizer=TfidfEmbeddings.tokenize, min_df=min_df)
        self.namespace_vectorizer = TfidfVectorizer(tokenizer=TfidfEmbeddings.tokenize, min_df=min_df)

    @staticmethod
    def split_into_subtokens(name: str):
        return [word.lower() for word in split(r'(?=[A-Z])', name) if word]

    @staticmethod
    def tokenize(doc: str):
        return (word.lower() for token in doc.split(".") for word in TfidfEmbeddings.split_into_subtokens(token))

    def fit(self, method_docs: List[str], namespace_docs: List[str]):
        self.method_vectorizer = self.method_vectorizer.fit(method_docs)
        self.namespace_vectorizer = self.namespace_vectorizer.fit(namespace_docs)

    def transform(self, frames: List[str]) -> torch.Tensor:
        tokens = [method.split(".") for method in frames]

        method_embeddings = self.method_vectorizer.transform([frame_tokens[-1] for frame_tokens in tokens]).todense()
        namespace_emdeddings = self.namespace_vectorizer.transform([".".join(frame_tokens[:-1]) for frame_tokens in tokens]).todense()

        return torch.cat((torch.Tensor(method_embeddings), torch.Tensor(namespace_emdeddings)), dim=-1)

    @property
    def n_embed(self):
        return len(self.method_vectorizer.vocabulary_) + len(self.namespace_vectorizer.vocabulary_)
import torch

from new.data.report import Frame
from new.model.frame_encoders.frame_encoder import FrameEncoder
from gensim.models import Word2Vec
import numpy as np

def clean_method_name(method_name):
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


class ScaffleFrameEncoder(FrameEncoder):
    def __init__(self, name):
        self.emb_model = Word2Vec.load(name)

    def encode(self, frame: Frame) -> torch.Tensor:
        method_name_tokens = clean_method_name(frame.name).split('.')
        embeddings = []
        for word in method_name_tokens:
            try:
                vector = self.emb_model.wv[word].reshape((self.dim(),))
            except KeyError:
                vector = np.zeros((self.dim(),))
            embeddings.append(vector)
        return torch.FloatTensor(embeddings)


    def dim(self) -> int:
        return self.emb_model.vector_size

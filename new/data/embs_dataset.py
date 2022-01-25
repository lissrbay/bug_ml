import os
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor

from typing import List, Union
import numpy as np
from new.data.report import Report


class EmbsDataset(Dataset):
    def __init__(self, report_ids: List[int], embs: Tensor):
        self.embs = embs
        self.report_ids = report_ids
        self.report_id_to_emb_pos = {id_: pos for pos, id_ in enumerate(self.report_ids)}
        self.dim = self.embs.shape[-1]

    def __len__(self):
        return len(self.report_ids)

    def __contains__(self, report_id):
        return report_id in self.report_id_to_emb_pos

    def __getitem__(self, report_id):
        emb_pos = self.report_idx_to_emb_pos[report_id]

        return self.embs[emb_pos]
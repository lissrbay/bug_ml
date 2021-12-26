from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, T_co

from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class ReportsDataModule(pl.LightningDataModule):
    def __init__(self, reports: List[Report], targets: List[List[int]], batch_size, encoder: ReportEncoder,
                 max_len: int, load_path: Optional[Path] = None):
        super().__init__()
        self.reports = reports
        self.targets = targets
        self.encoder = encoder
        self.batch_size = batch_size
        self.max_len = max_len
        self.load_path = load_path

    def setup(self, stage: Optional[str] = None):
        self.rtrain, self.rval = train_test_split(ReportsDataset(
            self.reports, self.targets, self.encoder, self.max_len, self.load_path), test_size=0.2, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rtrain, self.batch_size, collate_fn=second_dim_collate, num_workers=12)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.rval, self.batch_size, collate_fn=second_dim_collate, num_workers=12)


def second_dim_collate(data):
    features, labels, masks = zip(*data)
    return torch.cat(features, dim=1), torch.cat(labels, dim=1), torch.cat(masks, dim=1)


class ReportsDataset(Dataset):
    def __init__(self, reports: List[Report], targets: List[List[int]], encoder: ReportEncoder, max_len: int,
                 load_path: Optional[Path]) -> None:
        self.reports = reports
        self.targets = targets
        self.encoder = encoder
        self.max_len = max_len
        self.load_path = load_path

        if self.load_path:
            self.x = torch.Tensor(np.load(self.load_path / "X.npy"))
            self.y = torch.Tensor(np.load(self.load_path / "y.npy"))

            assert (self.y.shape[1] == max_len)

    def _encode_report(self, ind) -> Tuple[Tensor, Tensor]:
        encoded_report = self.encoder.encode_static(self.reports[ind])

        target_tensor = torch.LongTensor(self.targets[ind][:self.max_len])

        return encoded_report, target_tensor

    def __len__(self):
        if self.load_path:
            return len(self.x)
        return len(self.reports)

    def __getitem__(self, index):
        if self.load_path:
            feature = self.x[index]
            target = self.y[index]
        else:
            feature, target = self._encode_report(index)

        feature = feature[:self.max_len]
        target = target[:self.max_len]

        length = feature.shape[0]

        feature = pad(feature, (0, 0, 0, self.max_len - length)).unsqueeze(1)
        label = pad(target, (0, self.max_len - length)).unsqueeze(1)

        mask = (torch.arange(self.max_len) < length).unsqueeze(1)
        mask = mask * (label != 2)

        label = label * mask
        feature = feature * mask.unsqueeze(-1)

        return feature.float(), label.long(), mask


class SimpleDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        return self.reports[index], self.targets[index]

    def __len__(self):
        return len(self.reports)

    def __init__(self, reports: List[Report], targets: List[List[int]]):
        self.reports = reports
        self.targets = targets

def simple_collate(data):
    return zip(*data)

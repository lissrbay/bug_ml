from typing import List, Optional, Union

import attr
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from new.data.report import Report


class ReportsDataModule(pl.LightningDataModule):
    def __init__(self, reports: List[Report], targets: List[List[int]], batch_size: int, max_len: int):
        super().__init__()
        self.reports = reports
        self.targets = targets
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage: Optional[str] = None):
        self.rtrain, self.rval = train_test_split(ReportsDataset(
            self.reports, self.targets, self.max_len), test_size=0.2, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rtrain, self.batch_size, collate_fn=report_collate, num_workers=12)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.rval, self.batch_size, collate_fn=report_collate, num_workers=12)


def second_dim_collate(data):
    features, labels, masks = zip(*data)
    return torch.cat(features, dim=1), torch.cat(labels, dim=1), torch.cat(masks, dim=1)


def report_collate(data):
    reports, targets, masks = zip(*data)
    return reports, torch.cat(targets, dim=1), masks


class ReportsDataset(Dataset):
    def __init__(self, reports: List[Report], targets: List[List[int]], max_len: int):
        self.reports = reports
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, index):
        report, target = self.reports[index], self.targets[index]
        report = attr.evolve(report, frames=report.frames[:self.max_len])
        target = torch.LongTensor(target[:self.max_len])

        length = len(report.frames)

        target = pad(target, (0, self.max_len - length), value=2).unsqueeze(1)

        mask = (torch.arange(self.max_len) < length).unsqueeze(1)
        mask = mask * (target != 2)

        target = target * mask

        return report, target, mask

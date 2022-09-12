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
    def __init__(self, reports: List[Report], targets: List[List[int]], batch_size: int, max_len: int,
                 label_style: Optional[str]):
        super().__init__()
        self.reports = reports
        self.targets = targets
        self.batch_size = batch_size
        self.max_len = max_len
        self.label_style = label_style

    def setup(self, stage: Optional[str] = None):
        self.rtrain, self.rval = train_test_split(ReportsDataset(
            self.reports, self.targets, self.max_len, self.label_style), test_size=0.2, shuffle=False)
        # self.rval, self.rtest = train_test_split(self.rval, test_size=0.8, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rtrain, self.batch_size, collate_fn=report_collate, num_workers=0, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.rval, self.batch_size, collate_fn=report_collate, num_workers=0)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.rtest, self.batch_size, collate_fn=report_collate, num_workers=0)


def second_dim_collate(data):
    features, labels, masks = zip(*data)
    return torch.cat(features, dim=1), torch.cat(labels, dim=1), torch.cat(masks, dim=1)


def report_collate(data):
    reports, targets, masks = zip(*data)
    return reports, torch.cat(targets, dim=1), masks


class ReportsDataset(Dataset):
    def __init__(self, reports: List[Report], targets: List[List[int]], max_len: int, label_style: Optional[str]):
        self.reports = reports
        self.targets = targets
        self.max_len = max_len
        self.label_style = label_style

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, index):
        report, target = self.reports[index], self.targets[index]
        report = attr.evolve(report, frames=report.frames[:self.max_len])
        if self.label_style == "scaffle":
            target = torch.FloatTensor(target[:self.max_len])
        else:
            target = torch.LongTensor(target[:self.max_len])

        length = len(report.frames)

        target = pad(target, (0, self.max_len - length)).unsqueeze(1)

        mask = (torch.arange(self.max_len) < length).unsqueeze(1)
        mask = mask * (target != 2)

        target = target * mask

        return report, target, mask

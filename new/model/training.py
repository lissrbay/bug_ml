from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchcrf import CRF
from torchmetrics import Metric

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
        encoded_report = self.encoder.encode_report(self.reports[ind])

        target_tensor = torch.LongTensor(self.targets[ind][:self.max_len])

        return encoded_report, target_tensor

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, index):
        if self.load_path:
            feature = self.x[index]
            target = self.y[index]
        else:
            feature, target = self._encode_report(index)

        length = feature.shape[0]

        feature = pad(feature[:self.max_len], (0, 0, 0, self.max_len - length)).unsqueeze(1)
        label = pad(target[:self.max_len], (0, self.max_len - length)).unsqueeze(1)

        mask = (torch.arange(self.max_len) < length).unsqueeze(1)
        mask = mask * (label != 2)

        label = label * mask
        feature = feature * mask.unsqueeze(-1)

        return feature.float(), label.long(), mask


def precision(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(pred.sum().item(), 1)


def recall(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(labels.sum().item(), 1)


def metrics_to_str(metrics: Dict[str, torch.Tensor]):
    return " ".join(f"{k}: {v.item():.3f}" for k, v in metrics.items())


def get_label_scores(crf: CRF, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    n, batch_size, _ = emissions.size()

    scores = torch.zeros(emissions.shape, dtype=emissions.dtype, device=emissions.device)

    for j in range(n):
        scores[j] = crf.transitions[labels[j - 1], :].clone()

    scores = scores + emissions
    scores = nn.Softmax(dim=-1)(scores)

    return scores * mask.float().unsqueeze(-1)


class Precision(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp += torch.sum(preds[mask] * target[mask], dim=0)
        self.total += target[mask].sum().item()

    def compute(self):
        return round((self.tp.float() / max(self.total, 1)).item(), 3)


class Recall(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp += torch.sum(preds[mask] * target[mask], dim=0)
        self.total += preds[mask].sum().item()

    def compute(self):
        return round((self.tp.float() / max(self.total, 1)).item(), 3)


class TopkAccuracy(Metric):
    def __init__(self, k, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scores: torch.Tensor):
        preds = preds * mask.float()
        scores = scores * mask.float().unsqueeze(-1)
        target = target * mask
        # scores = scores * preds.unsqueeze(-1)
        scores = scores[:, :, 1]

        inds = scores.topk(self.k, dim=0).indices
        gathered = torch.gather(target, 0, inds).sum(0) > 0

        self.tp += torch.sum(gathered)
        self.total += torch.sum(torch.any(target, dim=0))

    def compute(self):
        return round((self.tp.float() / max(self.total, 1)).item(), 3)

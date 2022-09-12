from typing import Dict

import torch
from torchmetrics import Metric


def precision(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(pred.sum().item(), 1)


def recall(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    tp = torch.sum(pred[mask] * labels[mask], dim=0)
    return tp.sum(), max(labels.sum().item(), 1)


def metrics_to_str(metrics: Dict[str, torch.Tensor]):
    return " ".join(f"{k}: {v.item():.3f}" for k, v in metrics.items())


class Precision(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp_prec", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_prec", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp_prec += torch.sum(preds[mask] * target[mask], dim=0)
        self.total_prec += target[mask].sum().item()

    def compute(self):
        return round((self.tp_prec.float() / max(self.total_prec, 1)).item(), 3)


class Recall(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_recall", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp_recall += torch.sum(preds[mask] * target[mask], dim=0)
        self.total_recall += preds[mask].sum().item()

    def compute(self):
        return round((self.tp_recall.float() / max(self.total_recall, 1)).item(), 3)


class TopkAccuracy(Metric):
    def __init__(self, k, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp_acc", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total_acc", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scores: torch.Tensor):
        preds = preds * mask.float()
        scores = scores * mask.float().unsqueeze(-1)
        target = target * mask
        # scores = scores * preds.unsqueeze(-1)
        scores = scores[:, :, 1]

        inds = scores.topk(self.k, dim=0).indices
        gathered = torch.gather(target, 0, inds).sum(0) > 0

        # if torch.sum(gathered) > 2:
        #     print(scores)
        #     print(target)
        #     print(gathered)
        #     print(inds)

        self.tp_acc += torch.sum(gathered)
        self.total_acc += torch.sum(torch.any(target, dim=0))

    def compute(self):
        return round((self.tp_acc.float() / max(self.total_acc, 1)).item(), 3)

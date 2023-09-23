from typing import Dict

import torch
from torchmetrics import Metric
from copy import deepcopy
from typing import Any, Dict, Optional, Union
import numpy as np

import torch
from torch import Tensor
from torch.nn import ModuleList

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection



class BootStrapper(Metric):
    full_state_update: Optional[bool] = True

    def __init__(
        self,
        base_metric: Metric,
        num_bootstraps: int = 10,
        mean: bool = True,
        std: bool = True,
        quantile: Optional[Union[float, Tensor]] = None,
        raw: bool = False,
        sampling_strategy: str = "poisson",
        prefix: str = '',
        model_name: str = '',
        logs_save_path: str = '',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                "Expected base metric to be an instance of torchmetrics.Metric" f" but received {base_metric}"
            )

        self.metrics = ModuleList([deepcopy(base_metric) for _ in range(num_bootstraps)])
        self.num_bootstraps = num_bootstraps
        self.preds_vals = []
        self.targets_vals = []
        self.masks_vals = []
        self.scores_vals = []
        self.logs_save_path = logs_save_path
        self.mean = mean
        self.std = std
        self.quantile = quantile
        self.raw = raw
        self.prefix = prefix
        self.sampling_strategy = sampling_strategy

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scores: torch.Tensor, **kwargs) -> None:
        self.preds_vals.append(preds)
        self.targets_vals.append(target)
        self.masks_vals.append(mask)
        self.scores_vals.append(scores)

    def bootstrap(self, preds, target, masks, scores):
        bootstraped_metrics = []
        for i in range(10):
            bootstraped_ids = np.random.choice(preds.shape[1], preds.shape[1], replace=True)
            bd_preds, bd_targets, bd_masks, bd_scores = (preds[:, bootstraped_ids], target[:, bootstraped_ids], 
            masks[:, bootstraped_ids], scores[:, bootstraped_ids])

            for m in self.metrics:
                m.update(bd_preds, bd_targets, bd_masks, scores=bd_scores)
            computed_vals = torch.stack([m.compute() for m in self.metrics], dim=0)
            for m in self.metrics:
                m.reset()
            bootstraped_metrics.extend(torch.unsqueeze(computed_vals, dim=0))
        return torch.cat(bootstraped_metrics, dim=0)

    def compute(self) -> Dict[str, Tensor]:
        """Computes the bootstrapped metric values.

        Always returns a dict of tensors, which can contain the following keys: ``mean``, ``std``, ``quantile`` and
        ``raw`` depending on how the class was initialized.
        """
        computed_vals = self.bootstrap(torch.cat(self.preds_vals, dim=1), torch.cat(self.targets_vals, dim=1), 
        torch.cat(self.masks_vals, dim=1), torch.cat(self.scores_vals, dim=1))
        
        output_dict = {}
        if self.mean:
            output_dict[self.prefix + "_mean"] = computed_vals.mean(dim=0)
        if self.std:
            output_dict[self.prefix + "_std"] = computed_vals.std(dim=0)
        if self.quantile is not None:
            low_q = round(self.quantile/2, 3)
            high_q = round(1-self.quantile/2, 3)
            output_dict[self.prefix + f"quantile_{low_q}"] = torch.quantile(computed_vals, self.quantile/2, interpolation='lower')
            output_dict[self.prefix + f"quantile_{high_q}"] = torch.quantile(computed_vals, 1-self.quantile/2, interpolation='lower')

        self.preds_vals = []
        self.targets_vals = []
        self.masks_vals = []
        self.scores_vals = []
        return output_dict


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
        self.name = "Precision"

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp_prec += torch.sum(preds[mask] * target[mask], dim=0)
        self.total_prec += target[mask].sum().item()

    def compute(self):
        return self.tp_prec.float() / max(self.total_prec, 1)


class Recall(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.name = "Recall"

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
        self.tp_recall += torch.sum(preds[mask] * target[mask], dim=0)
        self.total_recall += preds[mask].sum().item()

    def compute(self):
        return self.tp_recall.float() / max(self.total_recall, 1)


class TopkAccuracy(Metric):
    def __init__(self, k, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp_acc", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total_acc", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k
        self.name = f"TopkAccuracy_{k}"

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scores: torch.Tensor):
        preds = preds * mask.float()
        scores = scores * mask.float().unsqueeze(-1)
        target = target * mask
        # scores = scores * preds.unsqueeze(-1)
        #print(target, scores, mask)
        scores = scores[:, :, 1]

        inds = scores.topk(self.k, dim=0).indices
        #print( inds.shape)
        gathered = torch.gather(target, 0, inds).sum(0) > 0

        # if torch.sum(gathered) > 2:
        #     print(scores)
        #     print(target)
        #     print(gathered)
        #     print(inds)

        self.tp_acc += torch.sum(gathered)
        self.total_acc += torch.sum(torch.any(target, dim=0))

    def compute(self):
        return (self.tp_acc.float() / max(self.total_acc, 1))

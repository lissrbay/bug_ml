from typing import Dict

import torch
from torchmetrics import Metric
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection


def _bootstrap_sampler(
    size: int,
    sampling_strategy: str = "multinomial",
) -> Tensor:
    if sampling_strategy == "poisson":
        p = torch.distributions.Poisson(1)
        n = p.sample((size,))
        return torch.arange(size).repeat_interleave(n.long(), dim=0)
    if sampling_strategy == "multinomial":
        idx = torch.multinomial(torch.ones(size), num_samples=size, replacement=True)
        return idx
    raise ValueError("Unknown sampling strategy")


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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                "Expected base metric to be an instance of torchmetrics.Metric" f" but received {base_metric}"
            )

        self.metrics = ModuleList([deepcopy(base_metric) for _ in range(num_bootstraps)])
        self.num_bootstraps = num_bootstraps

        self.mean = mean
        self.std = std
        self.quantile = quantile
        self.raw = raw
        self.prefix = prefix
        allowed_sampling = ("poisson", "multinomial")
        if sampling_strategy not in allowed_sampling:
            raise ValueError(
                f"Expected argument ``sampling_strategy`` to be one of {allowed_sampling}"
                f" but recieved {sampling_strategy}"
            )
        self.sampling_strategy = sampling_strategy

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Updates the state of the base metric.

        Any tensor passed in will be bootstrapped along dimension 0.
        """
        for idx in range(self.num_bootstraps):
            args_sizes = apply_to_collection(args, Tensor, len)
            kwargs_sizes = list(apply_to_collection(kwargs, Tensor, len))
            if len(args_sizes) > 0:
                size = args_sizes[0]
            elif len(kwargs_sizes) > 0:
                size = kwargs_sizes[0]
            else:
                raise ValueError("None of the input contained tensors, so could not determine the sampling size")
            sample_idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
            new_args = apply_to_collection(args, Tensor, torch.index_select, dim=0, index=sample_idx)
            new_kwargs = apply_to_collection(kwargs, Tensor, torch.index_select, dim=0, index=sample_idx)
            self.metrics[idx].update(*new_args, **new_kwargs)

    def compute(self) -> Dict[str, Tensor]:
        """Computes the bootstrapped metric values.

        Always returns a dict of tensors, which can contain the following keys: ``mean``, ``std``, ``quantile`` and
        ``raw`` depending on how the class was initialized.
        """
        computed_vals = torch.stack([m.compute() for m in self.metrics], dim=0)
        output_dict = {}
        if self.mean:
            output_dict[self.prefix + "_mean"] = computed_vals.mean(dim=0)
        if self.std:
            output_dict[self.prefix + "_std"] = computed_vals.std(dim=0)
        if self.quantile is not None:
            low_q = round(self.quantile/2, 3)
            high_q = round(1-self.quantile/2, 3)
            output_dict[f"quantile_{low_q}"] = torch.quantile(computed_vals, self.quantile/2, interpolation='lower')
            output_dict[f"quantile_{high_q}"] = torch.quantile(computed_vals, 1-self.quantile/2, interpolation='lower')

        #if self.raw:
        output_dict["raw"] = computed_vals
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
        return (self.tp_acc.float() / max(self.total_acc, 1))

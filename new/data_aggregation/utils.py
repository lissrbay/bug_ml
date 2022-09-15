import os
from dataclasses import dataclass
from typing import Optional

from torch import FloatTensor
from torch.nn.functional import pad
from tqdm import tqdm


@dataclass(frozen=True)
class MethodSignature:
    name: Optional[str] = None
    type: Optional[str] = None


def iterate_reports(path_to_reports: str, ext: str = ""):
    for root, _, files in filter(lambda x: x[0] == path_to_reports, os.walk(path_to_reports)):
        for file in tqdm(files):
            if os.path.splitext(file)[1] != ext:
                continue
            yield os.path.join(root, file)


def pad_features(report_features, frames_max):
    pad_size = frames_max - min(len(report_features), frames_max)
    return pad(FloatTensor(report_features), (0, 0, 0, pad_size))


def parse_method_signature(changed_methods):
    methods = []
    for changed_method in changed_methods:
        full_method_name = changed_method.name
        filename = full_method_name.split('::')[0]
        method = full_method_name.split('::')[-1].split(' ')[-1]
        methods.append((filename, method))

    return methods

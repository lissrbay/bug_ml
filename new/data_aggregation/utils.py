import os

from tqdm import tqdm
from torch.nn.functional import pad
from torch import FloatTensor


def iterate_reports(reports_dir: str, ext: str = ""):
    for root, _, files in filter(lambda x: x[0] == reports_dir, os.walk(reports_dir)):
        for file in tqdm(files):
            if os.path.splitext(file)[1] != ext:
                continue
            yield os.path.join(root, file)


def pad_features(report_features, frames_max):
    pad_size = frames_max - min(len(report_features), frames_max)
    return pad(FloatTensor(report_features), (0, 0, 0, pad_size))

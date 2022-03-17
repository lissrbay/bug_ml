import os
from tqdm import tqdm


def iterate_reports(path_to_reports: str, ext=""):
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            if os.path.splitext(file)[1] != ext:
                continue
            yield os.path.join(root, file)
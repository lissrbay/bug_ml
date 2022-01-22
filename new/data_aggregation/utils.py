import os
from tqdm import tqdm


def iterate_reports(path_to_reports: str, format=None):
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            if file[0] == '.':
                continue
            if format and os.path.splitext(file)[1] != format:
                continue
            yield file
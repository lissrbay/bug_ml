import json
import os
from tqdm import tqdm
from collections import Counter
from .add_path_info import load_report
path_to_reports = "//reports"

def add_count_labels_on_positions_in_report(report, counts):
    for i, frame in enumerate(report['frames']):
        if frame['label'] == 1:
            counts[i] += 1
            break
def count_labels_on_positions(path_to_reports):
    counts = [0 for i in range(100000)]
    mx = 0
    for root, _, files in os.walk(path_to_report):
        if not (root == path_to_report):
            continue
        for file in tqdm(files):
            report = load_report(path_to_report + "//" + file)
            mx = max(mx, len(report['frames']))
        add_count_labels_on_positions_in_report(report, counts)

    counts_without_zeros = [(j, i) for j, i in enumerate(counts) if i > 0]
    return counts_without_zeros

def count_quantiles(quantiles=[0.95, 0.99]):
    for quantile in quantiles:
        limit = 0.99*sum(counts)
        acc = 0
        x = 0
        for i, count in counts_:
            if acc + count < limit:
                acc += count
                x = i
        print("{} reports that is {} quantile had labels before {}-th frame".format(acc, quantile, x))
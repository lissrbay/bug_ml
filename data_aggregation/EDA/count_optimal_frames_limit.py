import json
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
path_to_reports = "/home/lissrbay/Рабочий стол/code2vec/code2vec_experiments/code2seq/labeled_reports"


def load_report(name):
    f = open(name, 'r')
    report = json.load(f)
    f.close()
    return report


def add_count_labels_on_positions_in_report(report, counts):
    for i, frame in enumerate(report['frames']):
        if frame['label'] == 1:
            counts[i] += 1
            break


def count_labels_on_positions(path_to_reports):
    counts = [0 for i in range(100000)]
    mx = 0
    for root, _, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            report = load_report(path_to_reports + "//" + file)
            mx = max(mx, len(report['frames']))
            add_count_labels_on_positions_in_report(report, counts)

    counts_without_zeros = [(j, i) for j, i in enumerate(counts) if i > 0]
    return counts_without_zeros

def count_quantiles(counts, quantiles=[0.95, 0.99]):
    for quantile in quantiles:
        limit = quantile*sum([i[1] for i in counts])
        acc = 0
        x = 0
        for i, count in counts:
            if acc + count < limit:
                acc += count
                x = i
        print("{} reports that is {} quantile had labels before {}-th frame".format(acc, quantile, x))


counts = count_labels_on_positions(path_to_reports)
all_values = sum([[j] * i for j, i in counts], [])
plt.hist(all_values, bins=80)
plt.show()
print(all_values)
count_quantiles(counts)
import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tqdm import tqdm


def load_report(name):
    try:
        f = open(name, 'r')
        report = json.load(f)
        f.close()
    except Exception:
        return {}
    return report


def add_count_labels_on_positions_in_report(report, counts):
    for i, frame in enumerate(report['frames']):
        if frame['label'] == 1:
            counts[i] += 1
            break
    return counts


def count_labels_on_positions(path_to_reports):
    counts = [0 for i in range(100000)]
    mx = 0
    for root, _, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            # print(os.path.join(path_to_reports, file))
            report = load_report(os.path.join(path_to_reports, file))
            if report == {}:
                continue
            mx = max(mx, len(report['frames']))
            counts = add_count_labels_on_positions_in_report(report, counts)

    counts_without_zeros = [(j, i) for j, i in enumerate(counts) if i > 0]
    return counts_without_zeros


def count_quantiles(counts, quantiles=[0.95, 0.99]):
    borders = []
    for quantile in quantiles:
        limit = quantile * sum([i[1] for i in counts])
        acc = 0
        x = 0
        for i, count in counts:
            if acc + count < limit:
                acc += count
                x = i
        print("{} reports that is {} quantile had labels before {}-th frame".format(acc, quantile, x))
        borders.append(x)

    return borders


def main(reports_path):
    reports_path = os.path.join(reports_path, 'labeled_reports')

    counts = count_labels_on_positions(reports_path)
    all_values = sum([[j] * i for j, i in counts], [])

    borders = count_quantiles(counts)
    all_values = list(filter(lambda x: x < borders[-1], all_values))
    plt.hist(all_values, bins=borders[-1])
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    args = parser.parse_args()
    reports_path = os.path.join(args.reports_path, 'labeled_reports')

    main(reports_path)

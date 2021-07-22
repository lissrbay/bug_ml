import argparse
import json
import os

import numpy as np
import pandas as pd


def load_report(name):
    f = open(name, 'r')
    report = json.load(f)
    f.close()
    return report


def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


def process_report(root):
    report_data = []
    method_with_bug = -1
    flag = 0
    report = load_report(root + '.json')
    for i, frame in enumerate(report['frames']):
        if frame['label']:
            method_with_bug = i
        method_name = clean_method_name(frame['method_name'])
        file_with_method = frame['file_name']
        if method_with_bug == i and not file_with_method:
            break
        print(file_with_method)
        if file_with_method:
            file_with_csv = os.path.join(root, file_with_method.split('.')[0] + '.csv')
            if method_with_bug == i and not os.path.exists(file_with_csv):
                break
            if os.path.exists(file_with_csv):
                df = pd.read_csv(file_with_csv)
                code_vector = df[df['method'] == method_name.lower()].values
                if code_vector.shape[0] > 0:
                    if method_with_bug == i:
                        flag = 1
                    report_data.append(code_vector[0, :])
                    continue
        report_data.append(np.zeros(384))
    return flag, report_data, method_with_bug


def process_data(path_to_files, path_to_methods):
    report_data = []
    report = load_report(path_to_methods)
    for i, method_info in enumerate(report[:80]):
        method_name = clean_method_name(method_info['method_name'])
        file_with_method = method_info['path']
        if file_with_method:
            file_with_csv = os.path.join(path_to_files, file_with_method.split('/')[-1].split('.')[0] + '.csv')
            if os.path.exists(file_with_csv):
                df = pd.read_csv(file_with_csv, index_col=0)
                code_vector = df[df['method'] == method_name.lower()].drop(['method'], axis=1).values
                if code_vector.shape[0] > 0:
                    report_data.append(code_vector[0, :])
                    continue
        report_data.append(np.zeros(320))
    for i in range(len(report_data), 80):
        report_data.append(np.zeros(320))
    return report_data


def match_embeddings_with_methods(path_to_report):
    data = []
    labels = []
    for root, dirs, files in os.walk(path_to_report):
        if not root.split('/')[-1].isnumeric():
            continue
        flag, report_data, method_with_bug = process_report(root)
        if flag:
            data.append(report_data)
            frames_labels = np.zeros(len(report_data))
            frames_labels[method_with_bug] = 1
            labels.append(frames_labels)
    return data, labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings_type", type=str)
    parser.add_argument("--reports_path", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_type_dir = args.embeddings_type
    reports_path = args.reports_path

    path_to_report = os.path.join(reports_path, "labeled_reports")
    data, labels = match_embeddings_with_methods(path_to_report)
    np.save(os.path.join("data", 'X(' + embeddings_type_dir + ')'), data)
    np.save(os.path.join("data", 'y(' + embeddings_type_dir + ')'), labels)


if __name__ == "__main__":
    main()

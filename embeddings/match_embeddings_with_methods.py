import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import sys
import pickle
from copy import deepcopy
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


def find_embedding_in_df(df, method_name):
    code_vectors = df[df['method'] == method_name.lower()].drop(['method'], axis=1).values
    if code_vectors.shape[0] > 0:
        return (True, code_vectors[0, :])
    return (False, None)


def process_report_by_file(root, frame_limit):
    report_data = []
    method_with_bug = -1
    report = load_report(root+'.json')
    frames_len = report['frames'][:frame_limit]

    for i, frame in enumerate(report['frames'][:frame_limit]):

        if frame['label']:
            method_with_bug = i
        method_name = clean_method_name(frame['method_name'])
        file_with_method = frame['file_name']
        if file_with_method:
            file_with_csv = os.path.join(root, file_with_method.split('.')[0] + '.csv')
            if os.path.exists(file_with_csv):
                df = pd.read_csv(file_with_csv, index_col=0)
                is_success, embedding = find_embedding_in_df(df, method_name)
                if is_success:
                    report_data.append(embedding)
                    continue
        report_data.append(np.zeros(320))

    for i in range(max(frame_limit - len(report_data), 0)):
        report_data.append(np.zeros(320))
    return frames_len, report_data, method_with_bug


def match_embeddings_with_methods(path_to_report, frame_limit):
    data = []
    labels = []
    report_ids = []
    for root, _, _ in tqdm(os.walk(path_to_report)):
        if not root.split('/')[-1].isnumeric():
            continue
        report_id = root.split('/')[-1]
        if not report_id:
            continue
        _, report_data, method_with_bug = process_report_by_file(root, frame_limit)
        if method_with_bug != -1:
            report_ids.append(report_id)
            data.append(report_data)
            frames_labels = np.zeros(len(report_data))
            frames_labels[method_with_bug] = 1
            labels.append(frames_labels)

    return data, labels, report_ids


def match_embeddings_with_methods_from_df(df, method_meta):
    if df is not None:
        is_success, embedding = find_embedding_in_df(df, method_meta['method_name'])
        if is_success:
            return embedding
    return np.zeros(320)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_type", type=str, default='code2seq')
    parser.add_argument("--reports_path", type=str, default='../intellij_fixed_201007')
    parser.add_argument("--frame_limit", type=int, default=80)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embeddings_type_dir = args.embeddings_type
    reports_path = args.reports_path
    frame_limit = args.frame_limit
    path_to_report = os.path.join(reports_path, "labeled_reports")
    data, labels, report_ids = match_embeddings_with_methods(path_to_report, frame_limit)
    np.save(os.path.join("..", "data", 'X(' + embeddings_type_dir + ')'), data) 
    np.save(os.path.join("..", "data", 'y(' + embeddings_type_dir + ')'), labels) 
    pickle.dump(report_ids, open(os.path.join('..', 'data', "reports_ids"), 'wb'))

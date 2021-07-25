import os
import json
import re
import pandas as pd
import numpy as np
import sys
import pickle
from copy import deepcopy

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
    code_vectors = df[df['method'] == method_name.lower()].values
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
            print(file_with_csv, os.path.exists(file_with_csv))

            if os.path.exists(file_with_csv):
                df = pd.read_csv(file_with_csv)
                is_success, embedding = find_embedding_in_df(df, method_name)
                if is_success:
                    report_data.append(embedding)
                    continue
        report_data.append(np.zeros(384))
    return frames_len, report_data, method_with_bug


def match_embeddings_with_methods(path_to_report, frame_limit):
    data = []
    labels = []
    report_ids = []
    print(path_to_report)
    for root, _, _ in os.walk(path_to_report):
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
    return np.zeros(384)


PATH_TO_REPORTS = os.path.join("..", "intellij_fixed_201007")
FILES_LIMIT = 80

if __name__ == "__main__":
    frame_limit = FILES_LIMIT
    embeddings_type_dir = "code2seq"
    path_to_report = PATH_TO_REPORTS
    if len(sys.argv) > 1:
        path_to_report = sys.argv[1]
        embeddings_type_dir = sys.argv[2]
        frame_limit = int(sys.argv[3])

    path_to_report = os.path.join(path_to_report, "labeled_reports")
    data, labels, report_ids = match_embeddings_with_methods(path_to_report, frame_limit)
    np.save(os.path.join("..", "data", 'X(' + embeddings_type_dir + ')'), data) 
    np.save(os.path.join("..", "data", 'y(' + embeddings_type_dir + ')'), labels) 
    pickle.dump(report_ids, open(os.path.join('..', 'data', "reports_ids"), 'wb'))

import os
import json
import re
import pandas as pd
import numpy as np
import sys
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

def process_report(root):
    report_data = []
    method_with_bug = -1
    flag = 0
    report = load_report(root+'.json')
    for i, frame in enumerate(report['frames']):
        if frame['label']:
            method_with_bug = i
        method_name = clean_method_name(frame['method_name'])
        file_with_method = frame['file_name']
        if method_with_bug == i and not file_with_method:
            break
        if file_with_method:
            file_with_csv = root + '/' + file_with_method.split('.')[0] + '.csv'
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

def pad_sequence(data, target, length):
  data_ = deepcopy(data)
  target_ = deepcopy(target)
  X = []
  for i, d in enumerate(data_):
    X_ = []
    for j in d:
      if j.shape[0] > 384:
        j = j[1:len(j)-1]
      X_.append(j)
    if len(d) > length:
      X_ = X_[:length]
      target_[i] = target_[i][:length]
    else:
      pad = length - len(d)
      size_of_vector = len(X_[0])
      for p in range(pad):
        curr_samples = len(X_)
        X_ = np.insert(np.array(X_).flatten(), np.array(X_).flatten().shape[0],
                        np.ones(size_of_vector)).reshape(curr_samples+1, 384)
        target_[i] = np.insert(target_[i], target_[i].shape[0], 0)
    X.append(X_)
  return np.asarray(X).astype('float32'), np.asarray(target_).astype('float32') 

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


if __name__ == "__main__":
    embeddings_type_dir = 'code2seq'
    frame_limit = 80

    if len(sys.argv) > 1:
        embeddings_type_dir = sys.argv[1]
    if len(sys.argv) > 2:
        frame_limit = int(sys.argv[2])

    path_to_report = embeddings_type_dir + "/labeled_reports"
    data, labels = match_embeddings_with_methods(path_to_report)
    data, target = pad_sequence(data, labels, 80)
    np.save('X(' + embeddings_type_dir + ')', data) 
    np.save('y(' + embeddings_type_dir + ')', target) 

import numpy as np
import os
import json
from tqdm import tqdm

def load_report(name):
    f = open(name, 'r')
    report = json.load(f)
    f.close()
    return report

reports_used = set(np.load('report_ids.npy'))
path_to_reports = "/home/lissrbay/Рабочий стол/code2vec/code2vec_experiments/code2seq/labeled_reports"
report_embeddings = []
report_labels = []
for root, dirs, files in os.walk(path_to_reports):
    if not (root == path_to_reports):
        continue
    for file in tqdm(files):
        report_id = file.split('.')[0]
        if not report_id in reports_used:
            report = load_report(path_to_reports + "//" + file)
            #print(report_id)
            bug_position = -1
            for i, frame in enumerate(report['frames'][:80]):
                if frame['label']:
                    bug_position = i
            if bug_position != -1:
                labels = np.zeros((80,))
                labels[bug_position] = 1
                report_labels.append(labels)
                embeddings = np.zeros((80, 320))
                report_embeddings.append(embeddings)
                
report_embeddings = np.array(report_embeddings)
report_labels = np.array(report_labels)
print(report_embeddings.shape)
np.save('unknown_embeddings.npy', report_embeddings)
np.save('unknown_embeddings_labels.npy', report_labels)
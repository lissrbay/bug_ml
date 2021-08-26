import json
import os
from tqdm import tqdm
import numpy as np
import sys

def baseline_accuracy(path_to_report):
    c = 0
    frames_count = []
    report_count = 0
    for _, _, files in os.walk(path_to_report):
        for file in files:
            if not file.endswith('.json'):
                continue 
            f = open(path_to_report + "//" + file, 'r')
            report = json.load(f)
            label_pos = -1
            report_count += 1
            for i, frame in enumerate(report['frames']):
                if frame['label']:
                    label_pos = i
            frames_count.append(len(report['frames']))
            if label_pos == 0:
                c += 1
    return c / report_count, np.array(frames_count).std()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PATH_TO_REPORTS = sys.argv[1]
    path_to_reports = os.path.join(PATH_TO_REPORTS, "labeled_reports")

    print(baseline_accuracy(path_to_reports))
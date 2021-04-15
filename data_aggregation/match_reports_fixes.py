import pandas as pd
import json
import os
from tqdm import tqdm
from add_path_info import load_report
import sys

def collect_info(path_to_reports):
    f = open("fixed_methods.txt", "r")
    fixed_methods = json.load(f)

    issues, method_names, path, hashes = [], [], [], []
    for issue, info in fixed_methods.items():
        for file, method in info['fixed_methods']:
            issues.append(int(issue))
            method_names.append(method)
            path.append(file)
            hashes.append(info['hash'])

    issues_info = pd.read_csv(os.path.join(path_to_reports, "issue_report_ids.csv"))
    issues_info_ = pd.DataFrame({"issue_id" : issues, "fixed_method" : method_names, 'path':path, 'hash' : hashes})
    full = issues_info.set_index("issue_id").join(issues_info_.set_index("issue_id"))
    full = full.dropna()
    full.to_csv("report_issue_methods.csv")

    return full 


def save_report(name, report):
    f = open(name, 'w')
    json.dump(report, f, indent=4)
    f.close()


def label_frame(frame, fixed_methods, paths):
    method = frame['method_name']
    frame['label'] = 0
    for i in range(len(fixed_methods)):
        fixed_method = fixed_methods[i]
        path = paths[i]
        if path != frame['file_name']:
            continue
        if method and fixed_method in method:
            frame['label'] = 1
            fixed_methods.pop(i)
            paths.pop(i)
            break
    return frame


def label_frames(report, methods_info):
    fixed_methods = []
    paths = []
    for info in methods_info.iterrows():
        info = info[1]
        fixed_methods.append(info['fixed_method'])
        paths.append(info['path'].split('/')[-1])
    for frame in report['frames']:
        frame = label_frame(frame, fixed_methods, paths)
            
    return report


def find_fixed_method_for_report(issues_info, report_id):
    fixed_methods = issues_info[issues_info['report_id'] == report_id]
    fixed_methods = fixed_methods[['fixed_method', 'path', 'hash']]

    return fixed_methods


def label_one_report(report, hash, fixed_methods, name):
    report['hash'] = hash
    report = label_frames(report, fixed_methods)
    save_report(name, report)


def label_reports(issues_info, path_to_report):
    for root, _, files in os.walk(path_to_report):
        if not (root == path_to_report):
            continue
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_report, file)
            report = load_report(path_to_file)
            report_id = report['id']

            fixed_methods = find_fixed_method_for_report(issues_info, report_id)
            if fixed_methods.shape[0] == 0:
                label_one_report(report, "", fixed_methods, path_to_file)
                continue
            else:
                hash = issues_info[issues_info['report_id'] == report_id]['hash'].values[0]
                label_one_report(report, hash, fixed_methods, path_to_file)

PATH_TO_REPORTS = os.path.join("..", "intellij_fixed_201007")
if __name__ == "__main__":
    path_to_report = PATH_TO_REPORTS
    if len(sys.argv) > 1:
        path_to_report = sys.argv[1]

    path_to_reports = os.path.join(path_to_report, "reports")
    issues_info = collect_info(path_to_reports)
    label_reports(issues_info, path_to_reports)
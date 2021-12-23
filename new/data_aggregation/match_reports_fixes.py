import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from new.data.report import Report, Frame
from typing import List, Dict


def collect_info(path_to_reports: str, data_dir: str = '../../data'):
    f = open(os.path.join(data_dir, "fixed_methods.txt"), "r")
    fixed_methods = json.load(f)

    issues, method_names, path, hashes = [], [], [], []
    for issue, info in fixed_methods.items():
        for file, method in info['fixed_methods']:
            issues.append(int(issue))
            method_names.append(method)
            path.append(file)
            hashes.append(info['hash'])

    issues_info = pd.read_csv(os.path.join(path_to_reports, "issue_report_ids.csv"))
    issues_info_ = pd.DataFrame({"issue_id": issues, "fixed_method": method_names, 'path': path, 'hash': hashes})
    full = issues_info.set_index("issue_id").join(issues_info_.set_index("issue_id"))
    full = full.dropna()
    full.to_csv(os.path.join(data_dir, "report_issue_methods.csv"))

    return full


def frame_label(frame: Frame, fixed_methods: List[str], paths: List[str]):
    method = frame.meta['method_name']
    for fixed_method, path in zip(fixed_methods, paths):
        if path != frame.meta['file_name']:
            continue
        if method and fixed_method in method:
            return 1
    return 0


def label_frames(report: Report, methods_info: pd.DataFrame):
    fixed_methods = methods_info.fixed_method.values
    paths = methods_info.path.apply(lambda x: x.split('/')[-1])

    for frame in report.frames:
        label = frame_label(frame, fixed_methods, paths)
        frame.meta['label'] = label

    return report


def find_fixed_method_for_report(issues_info: pd.DataFrame, report_id: int):
    fixed_methods = issues_info[issues_info['report_id'] == report_id]
    fixed_methods = fixed_methods[['fixed_method', 'path', 'hash']]

    return fixed_methods


def label_one_report(report: Report, fixed_methods: pd.DataFrame):
    return label_frames(report, fixed_methods)


def get_hash(report_id: int, issues_info: pd.DataFrame):
    return issues_info[issues_info['report_id'] == report_id]['hash'].values[0]


def label_reports(issues_info: pd.DataFrame, path_to_reports: str):
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = Report.load_from_base_report(path_to_file)

            fixed_methods = find_fixed_method_for_report(issues_info, report.id)
            hash = ""
            if fixed_methods.shape[0] != 0:
                hash = get_hash(report.id, issues_info)
                report.hash = hash
            label_one_report(report, hash, fixed_methods)

            if report.id != 0:
                print(os.path.join('../..', path_to_reports, 'tmp', file.split('.')[0]))
                report.save_report(os.path.join(path_to_reports, 'tmp', file.split('.')[0]))


def main(reports_path: pd.DataFrame):
    path_to_reports = os.path.join(reports_path, "reports")
    issues_info = collect_info(reports_path)
    label_reports(issues_info, path_to_reports)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    args = parser.parse_args()
    main(args.reports_path)

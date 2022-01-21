import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from new.data.report import Report, Frame
from typing import List, Dict
from new.data_aggregation.collect_reports_info import INTELLIJ_CHANGED_METHODS_FILE, ISSUE_REPORTS_MAPPING_FILE


def collect_info(path_to_reports: str, data_dir: str) -> pd.DataFrame:
    issues, method_names, path, hashes = [], [], [], []

    with open(os.path.join(data_dir, INTELLIJ_CHANGED_METHODS_FILE), "r") as fixed_methods_io:
        fixed_methods = json.load(fixed_methods_io)

        for issue, info in fixed_methods.items():
            for file, method in info['fixed_methods']:
                issues.append(int(issue))
                method_names.append(method)
                path.append(file)
                hashes.append(info['hash'])

    issues_info = pd.read_csv(os.path.join(path_to_reports, ISSUE_REPORTS_MAPPING_FILE))
    issues_info_ = pd.DataFrame({"issue_id": issues, "fixed_method": method_names, 'path': path, 'hash': hashes})
    full = issues_info.set_index("issue_id").join(issues_info_.set_index("issue_id"))
    full = full.dropna()

    return full


def frame_label(frame: Frame, fixed_methods: List[str], paths: List[str]) -> int:
    method = frame.meta['method_name']
    for fixed_method, path in zip(fixed_methods, paths):
        if path != frame.meta['file_name']:
            continue
        if method and fixed_method in method:
            return 1
    return 0


def label_frames(report: Report, methods_info: pd.DataFrame) -> Report:
    fixed_methods = methods_info.fixed_method.values
    paths = methods_info.path.apply(lambda x: x.split('/')[-1])

    frames_with_labels = []
    for frame in report.frames:
        label = frame_label(frame, fixed_methods, paths)
        frame_meta = frame.meta
        frame_meta['label'] = label
        frames_with_labels.append(Frame(frame.code, frame_meta))

    return Report(report.id, report.exceptions, report.hash, frames_with_labels)


def find_fixed_method_for_report(issues_info: pd.DataFrame, report_id: int) -> pd.DataFrame:
    fixed_methods = issues_info[issues_info['report_id'] == report_id]
    fixed_methods = fixed_methods[['fixed_method', 'path', 'hash']]

    return fixed_methods


def get_hash(report_id: int, issues_info: pd.DataFrame) -> str:
    return issues_info[issues_info['report_id'] == report_id]['hash'].values[0]


def label_reports(issues_info: pd.DataFrame, path_to_reports: str):
    reports_success = 0
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = Report.load_from_base_report(path_to_file)

            fixed_methods = find_fixed_method_for_report(issues_info, report.id)
            hash = ""
            if fixed_methods.shape[0] != 0:
                hash = get_hash(report.id, issues_info)
                report.hash = hash

            report = label_frames(report, hash, fixed_methods)

            reports_success += 1 if sum([frame.meta['label'] for frame in report]) else 0
            if report.id != 0:
                report.save_report(os.path.join(path_to_reports, 'tmp', file.split('.')[0]))

    print(f"Successed label data for {reports_success} reports.")


def match_reports_to_labels(raw_reports_path: pd.DataFrame, data_dir:str = '../../data'):
    path_to_reports = os.path.join(raw_reports_path, "labeled_reports")
    issues_info = collect_info(raw_reports_path, data_dir)
    label_reports(issues_info, path_to_reports)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    args = parser.parse_args()
    match_reports_to_labels(args.reports_path)

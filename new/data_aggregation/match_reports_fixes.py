import json
import os
from argparse import ArgumentParser
from typing import List

import pandas as pd

from new.constants import REPO_CHANGED_METHODS_FILE, ISSUE_REPORTS_MAPPING_FILE, REPORTS_INTERMEDIATE_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports


def collect_info(path_to_reports: str, data_dir: str) -> pd.DataFrame:
    issues, method_names, path, hashes = [], [], [], []

    with open(os.path.join(data_dir, REPO_CHANGED_METHODS_FILE), "r") as fixed_methods_io:
        fixed_methods = json.load(fixed_methods_io)

        for issue, info in fixed_methods.items():
            for changed_method_info in info['fixed_methods']:
                file_path = changed_method_info['path']
                method = changed_method_info['name']
                issues.append(int(issue))
                method_names.append(method)
                path.append(file_path)
                hashes.append(info['hash'])

    issues_reports = pd.read_csv(os.path.join(path_to_reports, ISSUE_REPORTS_MAPPING_FILE))
    issues_fixed_methods = pd.DataFrame(
        {"issue_id": issues, "fixed_method": method_names, 'path': path, 'hash': hashes}
    )
    full = issues_reports.set_index("issue_id").join(issues_fixed_methods.set_index("issue_id"), how='inner')
    full = full.dropna()

    return full


def label_frames(report: Report, commit_hash: str, methods_info: pd.DataFrame):
    fixed_methods = methods_info.fixed_method.values
    paths = methods_info.path.apply(lambda x: x.split('/')[-1])

    for frame in report.frames:
        method = frame.method_name
        for fixed_method, path in zip(fixed_methods, paths):
            if method and fixed_method in method:
                frame.label = 1


def find_fixed_method_for_report(issues_info: pd.DataFrame, report_id: int) -> pd.DataFrame:
    fixed_methods = issues_info[issues_info['report_id'] == report_id]
    fixed_methods = fixed_methods[['fixed_method', 'path', 'hash']]

    return fixed_methods


def get_hash(report_id: int, issues_info: pd.DataFrame) -> str:
    return issues_info[issues_info['report_id'] == report_id]['hash'].values[0]


def label_reports(issues_info: pd.DataFrame, path_to_reports: str, path_to_reports_save: str):
    reports_success = 0
    for file_name in iterate_reports(path_to_reports, ext='.json'):
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_from_base_report(path_to_file)

        fixed_methods = find_fixed_method_for_report(issues_info, report.id)
        report_hash = report.hash
        if fixed_methods.shape[0] != 0:
            report_hash = get_hash(report.id, issues_info)

        label_frames(report, report_hash, fixed_methods)
        report_success = 1 if sum([frame.label for frame in report.frames]) else 0
        reports_success += report_success
        if report.id != 0 and report_success:
            report.save_report(os.path.join(path_to_reports_save, str(report.id)))

    print(f"Successed label data for {reports_success} reports.")


def match_reports_to_labels(raw_reports_path: str, data_dir: str = '../../data'):
    path_to_reports = os.path.join(raw_reports_path, "reports")
    reports_save_path = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    os.makedirs(reports_save_path, exist_ok=True)
    issues_info = collect_info(raw_reports_path, data_dir)
    label_reports(issues_info, path_to_reports, reports_save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    args = parser.parse_args()
    match_reports_to_labels(args.reports_path)

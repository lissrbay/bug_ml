import json
from argparse import ArgumentParser
from typing import Set, List

from new.data_aggregation.utils import iterate_reports
import os
from data.report import Report, Frame
from new.constants import REPO_CHANGED_METHODS_FILE, ISSUE_REPORTS_MAPPING_FILE, SCAFFLE_REPORTS_DIR
from new.model.frame_encoders.scaffle import clean_method_name
import re
import pandas as pd


def get_method_tokens(path: str, method_name: str):
    method_name = clean_method_name(method_name)
    method_name_tokens = tokenize(method_name)

    file_name = tokenize(path)
    method_tokens = set(file_name)
    if len(method_name) > 0:
        method_tokens.update(set(method_name_tokens))
    return method_tokens


def frame_label(frame: Frame, fixed_methods: List[Set[str]]) -> int:
    method_tokens = get_method_tokens(frame.meta['path'], frame.meta['method_name'])
    max_s = 0.0
    for fixed_path in fixed_methods:
        s = len(fixed_path.intersection(method_tokens))/len(fixed_path) if len(fixed_path) > 0 else 0
        if s > max_s:
            max_s = s

    return max_s


def label_frames(report: Report, methods_info: pd.DataFrame) -> Report:
    fixed_methods = methods_info.fixed_method.values[0]

    frames_with_labels = []
    for frame in report.frames:
        label = frame_label(frame, fixed_methods)
        frame_meta = frame.meta
        frame_meta['ground_truth'] = frame.meta['label']
        frame_meta['label'] = label
        frames_with_labels.append(Frame(frame.code, frame_meta))
    return Report(report.id, report.exceptions, report.hash, frames_with_labels)


def tokenize(s: str) -> str:
    return re.split('/|\.', s)


def match_fixed_methods_tokens(path_to_reports: str, data_dir: str) -> pd.DataFrame:
    issues, method_names = [], []

    with open(os.path.join(data_dir, REPO_CHANGED_METHODS_FILE), "r") as fixed_methods_io:
        fixed_methods = json.load(fixed_methods_io)
        for issue, info in fixed_methods.items():
            issues.append(int(issue))
            issue_fixed_methods = []
            for changed_method_info in info['fixed_methods']:
                method_tokens = get_method_tokens(changed_method_info['path'], '')
                issue_fixed_methods.append(method_tokens)
            method_names.append(issue_fixed_methods)

    issues_reports = pd.read_csv(os.path.join(path_to_reports, ISSUE_REPORTS_MAPPING_FILE))
    issues_fixed_methods = pd.DataFrame(
        {"issue_id": issues, "fixed_method": method_names}
    )
    issues_info = issues_reports.set_index("issue_id").join(issues_fixed_methods.set_index("issue_id"), how='inner')
    issues_info = issues_info.dropna()
    return issues_info


def scaffle_labeling(path_to_reports: str, data_dir: str):
    issues_info = match_fixed_methods_tokens(path_to_reports, data_dir)

    reports_success = 0
    for file_name in iterate_reports(path_to_reports):
        report_success = 0
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_report(path_to_file)
        fixed_methods = issues_info[issues_info['report_id'] == report.id]
        if fixed_methods.shape[0] > 0:
            report = label_frames(report, fixed_methods)
            report_success = 1

        if report.id != 0 and report_success:
            reports_success += 1
            report.save_report(os.path.join(data_dir, SCAFFLE_REPORTS_DIR, str(report.id)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    scaffle_labeling(args.reports_path, args.data_dir)
import argparse
import os
from copy import deepcopy
from os.path import join, exists
from typing import Iterable, List, Dict, Set

import pandas as pd
from tqdm import tqdm

from new.data.report import Report, Frame


def load_report(report_id: int, reports_dir: str) -> Report:
    file_path = join(reports_dir, f"{report_id}.json")
    return Report.from_json(file_path)


def load_methods(issue_id: int, methods_dir: str, sep: str = ".") -> List[str]:
    file_path = join(methods_dir, f"{issue_id}.txt")

    if not exists(file_path):
        return []

    with open(file_path) as file:
        method_names = file.readlines()

    method_names = [name.strip().replace("/", sep) for name in method_names]
    return method_names


def load_report_to_issue(file_path: str) -> Dict[int, int]:
    df = pd.read_csv(file_path)

    report_to_issue = {}
    for _, row in df.iterrows():
        report_to_issue[int(row["report_id"])] = int(row["issue_id"])

    return report_to_issue


def get_reports(reports_dir: str) -> Iterable[Report]:
    report_ids = [int(rid[:-5]) for rid in os.listdir(reports_dir)]
    for report_id in report_ids:
        yield load_report(report_id, reports_dir)


def label_frame(frame: Frame, tokenized_names: List[Set[str]]):
    # Inplace
    # Remove lambdas
    method_tokenized = set()
    for name in frame.method_name.split("."):
        lambda_pos = name.find("$")
        if lambda_pos > 0:
            method_tokenized.add(name[:lambda_pos])
        else:
            method_tokenized.add(name)

    max_score = 0.

    for name in tokenized_names:
        intersection = method_tokenized & name
        score = len(intersection) / len(method_tokenized)
        max_score = max(max_score, score)

    frame.label = round(max_score, 2)


def label_report(report: Report, method_names: List[str]) -> "Report":
    labeled_report = deepcopy(report)
    tokenized_names = [set(name.split(".")) for name in method_names]

    for frame in labeled_report.frames:
        label_frame(frame, tokenized_names)

    return labeled_report


def label_reports(reports_dir: str, issue_report_path: str, data_dir: str):
    """
    Labels report frames and saves to data directory.
    """

    print("Labeling reports")

    report_to_issue = load_report_to_issue(issue_report_path)
    for report in tqdm(get_reports(reports_dir)):
        issue_id = report_to_issue[report.id]
        changed_methods = load_methods(issue_id, join(data_dir, "changed_methods"), sep=".")
        labeled_report = label_report(report, changed_methods)
        labeled_report.save_json(join(data_dir, "reports"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", default="/Users/Denis.Sushentsev/Work/intellij_fixed_201007_raw/reports",
                        type=str)
    parser.add_argument("--issue_report_path", default="/Users/Denis.Sushentsev/Work/intellij_fixed_201007_raw/issue_report_ids.csv",
                        type=str)
    parser.add_argument("--data_dir", default="/Users/Denis.Sushentsev/Study/bug_ml_data",
                        type=str)
    args = parser.parse_args()

    label_reports(args.reports_dir, args.issue_report_path, args.data_dir)

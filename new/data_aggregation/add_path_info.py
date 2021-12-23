import argparse
import json
import os
from collections import defaultdict

from git import Repo, db
from tqdm import tqdm
from new.data.report import Report, Frame
from typing import List, Dict


def list_files_in_commit(commit: str, repo: Repo):
    configFiles = repo.git.execute(
        ['git', 'ls-tree', '-r', '--name-only', commit.hexsha]).split()
    java_files = defaultdict(list)
    for file in configFiles:
        if file.endswith('.java') or file.endswith('.kt'):
            java_files[file.split('/')[-1]].append(file)

    return java_files


def get_method_path_and_name(frame: Frame):
    path = frame.meta['method_name']
    path = path.split('$')[0]
    path = path.split('.')
    method = path.pop()
    path = "/".join(path)
    return path, method


def find_file_for_frame(frame: Frame, matching_files: List[str]):
    frame.meta['path'] = ""
    for file_path in matching_files:
        if frame.meta['file_name'] in file_path:
            frame.meta['path'] = file_path
            break

    return frame


def add_paths_to_report(report: Report, commit_files:List[List[str]], file_limit: int):
    for frame in report.frames[:80]:
        matching_files_for_frame = commit_files[frame.meta['file_name']]
        find_file_for_frame(frame, matching_files_for_frame)

    return report


def add_paths_to_all_reports(from_repo: Repo, path_to_reports: str, path_to_reports_save: str, file_limit=80):
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = Report.load_report(path_to_file)
            if report == {}:
                continue
            commit = from_repo.commit(report.hash + '~1')
            commit_files = list_files_in_commit(commit, from_repo)

            add_paths_to_report(report, commit_files, file_limit=file_limit)
            report.save_report(os.path.join(path_to_reports_save, file))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--intellij_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


def main(intellij_path: str, reports_path: str, files_limit: int = 80):
    repo = Repo(intellij_path, odbt=db.GitDB)

    path_to_reports = os.path.join(reports_path, "reports", "tmp")
    path_to_reports_save = os.path.join(reports_path, "labeled_reports")

    add_paths_to_all_reports(repo, path_to_reports, path_to_reports_save, files_limit)


if __name__ == "__main__":
    args = parse_args()

    intellij_path = args.intellij_path
    reports_path = args.reports_path
    files_limit = args.files_limit
    main(intellij_path, reports_path, files_limit)

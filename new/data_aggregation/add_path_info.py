import argparse
import os
from collections import defaultdict
from typing import List, Tuple, Dict

from git import Repo, db, Commit

from new.constants import REPORTS_INTERMEDIATE_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports


def list_files_in_commit(commit: Commit, repo: Repo) -> Dict[str, List[str]]:
    config_files = repo.git.execute(
        ['git', 'ls-tree', '-r', '--name-only', commit.hexsha]).split()
    java_files = defaultdict(list)
    for file in config_files:
        if file.endswith('.java') or file.endswith('.kt'):
            java_files[file.split('/')[-1]].append(file)

    return java_files


def get_method_path_and_name(frame: Frame) -> Tuple[str, str]:
    path = frame.method_name
    path = path.split('$')[0]
    path = path.split('.')
    method = path.pop()
    path = "/".join(path)
    return path, method


def add_paths_to_all_reports(from_repo: Repo, path_to_reports: str, path_to_reports_save: str):
    reports_success = 0
    for file_name in iterate_reports(path_to_reports):
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_report(path_to_file)
        if report.id == 0:
            continue
        try:
            commit = from_repo.commit(report.hash + '~1')
            commit_files = list_files_in_commit(commit, from_repo)
        except Exception:
            continue

        for frame in report.frames:
            matching_files_for_frame = commit_files[frame.file_name]
            for file_path in matching_files_for_frame:
                if frame.file_name in file_path:
                    frame.path = file_path

        report.save_report(os.path.join(path_to_reports_save, file_name))
        reports_success += 1

    print(f"Successed add paths for {reports_success} reports.")


def add_paths_to_reports(repo_path: str, data_dir: str):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    reports_save_path = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)

    add_paths_to_all_reports(repo, path_to_reports, reports_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    # parser.add_argument("--files_limit", type=int, default=80)

    args = parser.parse_args()

    add_paths_to_reports(args.repo_path, args.reports_path)

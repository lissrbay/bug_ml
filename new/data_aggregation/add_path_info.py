import argparse
import os
from collections import defaultdict
from typing import List, Tuple, Dict

from git import Repo, db, Commit

from new.constants import REPORTS_INTERMEDIATE_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports


def list_files_in_commit(commit: Commit, repo: Repo) -> Dict[str, List[str]]:
    configFiles = repo.git.execute(
        ['git', 'ls-tree', '-r', '--name-only', commit.hexsha]).split()
    java_files = defaultdict(list)
    for file in configFiles:
        if file.endswith('.java') or file.endswith('.kt'):
            java_files[file.split('/')[-1]].append(file)

    return java_files


def get_method_path_and_name(frame: Frame) -> Tuple[str, str]:
    path = frame.meta['method_name']
    path = path.split('$')[0]
    path = path.split('.')
    method = path.pop()
    path = "/".join(path)
    return path, method


def find_file_for_frame(frame: Frame, matching_files: List[str]) -> str:
    for file_path in matching_files:
        if frame.meta['file_name'] in file_path:
            return file_path

    return ""


def add_paths_to_one_report(report: Report, commit_files: Dict[str, List[str]], file_limit: int) -> Report:
    frames_with_paths = []
    for frame in report.frames[:file_limit]:
        matching_files_for_frame = commit_files[frame.meta['file_name']]
        frame_path = find_file_for_frame(frame, matching_files_for_frame)
        frame_meta = frame.meta
        frame_meta['path'] = frame_path
        frames_with_paths.append(Frame(frame.code, frame_meta))

    return Report(report.id, report.exceptions, report.hash, frames_with_paths)


def add_paths_to_all_reports(from_repo: Repo, path_to_reports: str, path_to_reports_save: str, file_limit: int = 80):
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

        report = add_paths_to_one_report(report, commit_files, file_limit=file_limit)
        report.save_report(os.path.join(path_to_reports_save, file_name))
        reports_success += 1

    print(f"Successed add paths for {reports_success} reports.")


def add_paths_to_reports(repo_path: str, data_dir: str, files_limit: int = 80):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    reports_save_path = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)

    add_paths_to_all_reports(repo, path_to_reports, reports_save_path, files_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    args = parser.parse_args()

    add_paths_to_reports(args.repo_path, args.reports_path, args.files_limit)

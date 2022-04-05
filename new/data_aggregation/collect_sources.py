import argparse
import base64
import os
import re

from git import Repo, db

from new.constants import REPORTS_INTERMEDIATE_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports


def remove_header(code: str) -> str:
    bias = 0
    source_code_lines = code.split("\n")
    for line in source_code_lines:
        if not re.search('^import', line.strip()):
            break
        bias += 1
    clean_code = source_code_lines[bias:]
    return '\n'.join(clean_code)


def get_file_by_commit(repo: Repo, commit_hash: str, diff_file: str) -> str:
    code = repo.git.show(f"{commit_hash}:{diff_file}")
    return remove_header(code)


def get_report_sources(repo: Repo, report: Report, commit_hash: str, top_frames: int) -> Report:
    frames_with_code = []
    for frame in report.frames[:top_frames]:
        if frame.meta['path'] != '':
            diff_file = frame.meta['path']
            try:
                code = get_file_by_commit(repo, commit_hash + "~1", diff_file)
                frame_source_code = base64.b64encode(code.encode('UTF-8'))
                frames_with_code.append(Frame(frame_source_code, frame.meta))
            except Exception:
                print(report.id, frame.meta['file_name'])

    return Report(report.id, report.exceptions, report.hash, frames_with_code)


def collect_all_reports_sources(repo: Repo, reports_dir: str, top_frames: int = 80):
    reports_success = 0
    for file_name in iterate_reports(reports_dir):
        report_path = os.path.join(reports_dir, file_name)
        report = Report.load_report(report_path)
        if report.hash == "":
            continue

        report = get_report_sources(repo, report, report.hash, top_frames)
        report.save_report(report_path)
        reports_success += 1

    print(f"Successfully collected code data for {reports_success} reports.")


def collect_reports_sources(repo_path: str, data_dir: str, top_frames: int):
    repo = Repo(repo_path, odbt=db.GitDB)
    reports_dir = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    collect_all_reports_sources(repo, reports_dir, top_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--top_frames", type=int, default=80)
    args = parser.parse_args()

    collect_reports_sources(args.repo_path, args.data_dir, args.top_frames)

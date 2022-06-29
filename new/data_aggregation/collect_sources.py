import argparse
import os
import re
import base64

from git import Repo, db

from new.constants import REPORTS_INTERMEDIATE_DIR
from new.data_aggregation.utils import iterate_reports
from new.data.report import Report, Frame


def remove_unused_begin(code: str) -> str:
    bias = 0
    for line in code.split('\n'):
        bias += 1
        if re.search('^import', line):
            break
    clean_code = code.split('\n')[bias:]
    return '\n'.join(clean_code)


def get_file_by_commit(repo: Repo, commit: str, diff_file: str) -> str:
    code = repo.git.show('{}:{}'.format(commit, diff_file))
    return code


def get_sources_for_report(repo: Repo, report: Report, commit: str, file_limit: int) -> Report:
    frames_with_codes = []
    for frame in report.frames[:file_limit]:
        if frame.meta['path'] != '':
            diff_file = frame.meta['path']
            try:
                code = get_file_by_commit(repo, commit + "~1", diff_file)
                hashed_code = base64.b64encode(code.encode('UTF-8'))
                frame_code = hashed_code
                frames_with_codes.append(Frame(frame_code, frame.meta))
            except Exception:
                print(report.id, frame.meta['file_name'])
        else:
            frames_with_codes.append(frame)

    return Report(report.id, report.exceptions, report.hash, frames_with_codes)


def collect_sources_for_all_reports(repo: Repo, path_to_reports: str, file_limit=80):
    reports_success = 0
    for file_name in iterate_reports(path_to_reports):
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_report(path_to_file)
        if report.hash == "":
            continue

        report = get_sources_for_report(repo, report, report.hash, file_limit)
        report.save_report(path_to_file)
        reports_success += 1

    print(f"Successed collect code data for {reports_success} reports.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


def collect_sources_for_reports(repo_path: str, data_dir: str, files_limit: int):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    collect_sources_for_all_reports(repo, path_to_reports, files_limit)


if __name__ == "__main__":
    args = parse_args()
    collect_sources_for_reports(args.repo_path, args.data_dir, args.files_limit)

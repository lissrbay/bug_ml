import argparse
import os
import re
import base64

from git import Repo, db
from tqdm import tqdm
from new.data.report import Report


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
    return remove_unused_begin(code)


def is_labeled_inside_window(report: Report, file_limit: int) -> int:
    flag = 0
    for frame in report.frames[:file_limit]:
        if frame.meta['file_name'] and frame.meta['label']:
            flag = 1
    return flag


def get_sources_for_report(repo: Repo, report: Report, commit: str, full_save_path: str, file_limit: int) -> Report:
    for frame in report.frames[:file_limit]:
        if frame.meta['path'] != '':
            diff_file = frame.meta['path']
            try:
                code = get_file_by_commit(repo, commit + "~1", diff_file)
                hashed_code = base64.b64encode(code.encode('UTF-8'))
                frame.code = hashed_code
            except Exception:
                print(os.path.join(full_save_path, frame.meta['file_name']))
    return report


def create_subfolder(path: str):
    try:
        os.mkdir(path)
    except Exception:
        # dir already exist
        pass


def collect_sources_for_reports(repo: Repo, save_path: str, path_to_reports: str, file_limit=80):
    reports_success = 0
    for root, _, files in filter(lambda x: (x[0] == path_to_reports), os.walk(path_to_reports)):
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = Report.load_report(path_to_file)
            if report.hash == "":
                continue

            full_save_path = os.path.join(save_path, str(report.id))
            create_subfolder(full_save_path)

            flag = is_labeled_inside_window(report, file_limit)
            if not flag:
                continue
            reports_success += 1
            report = get_sources_for_report(repo, report, report.hash, full_save_path, file_limit)
            report.save_report(path_to_file)

    print(f"Successed collect code data for {reports_success} reports.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--intellij_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


def main(intellij_path: str, reports_path: str, files_limit: int):
    repo = Repo(intellij_path, odbt=db.GitDB)

    path_to_reports = os.path.join(reports_path, "labeled_reports")
    save_path = os.path.join(reports_path, "labeled_reports")
    collect_sources_for_reports(repo, save_path, path_to_reports, files_limit)


if __name__ == "__main__":
    args = parse_args()
    intellij_path = args.intellij_path
    reports_path = args.reports_path
    files_limit = args.files_limit
    main(intellij_path, reports_path, files_limit)

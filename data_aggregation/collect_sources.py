import argparse
import os
import re

from git import Repo, db
from tqdm import tqdm

from add_path_info import load_report, create_subfolder


def remove_unused_begin(code):
    bias = 0
    for line in code.split('\n'):
        bias += 1
        if re.search('^import', line):
            break
    clean_code = code.split('\n')[bias:]
    return '\n'.join(clean_code)


def get_file_by_commit(repo, commit, diff_file):
    code = repo.git.show('{}:{}'.format(commit, diff_file))
    return remove_unused_begin(code)


def is_labeled_inside_window(report, file_limit):
    flag = 0
    for i, frame in enumerate(report['frames']):
        if i == file_limit:
            break
        if frame['file_name'] and frame['label']:
            flag = 1
    return flag


def get_sources_for_report(repo, report, commit, full_save_path, file_limit=80):
    for i, frame in enumerate(report['frames'][:file_limit]):
        if i == file_limit:
            break
        if frame['path'] != '':
            diff_file = frame['path']
            code = get_file_by_commit(repo, commit + "~1", diff_file)
            f = open(os.path.join(full_save_path, frame['file_name']), 'w', encoding="utf-8")
            f.write(code)
            f.close()


def get_sources_for_path(repo, path, full_save_path):
    code = get_file_by_commit(repo, "HEAD", path)
    f = open(os.path.join(full_save_path, path), 'w', encoding="utf-8")
    f.write(code)
    f.close()


def collect_sources_for_reports(repo, save_path, path_to_reports, file_limit=80):
    for root, dirs, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = load_report(path_to_file)
            commit = report['hash']
            if commit == "":
                continue
            full_save_path = os.path.join(save_path, str(report['id']))
            create_subfolder(full_save_path)

            flag = is_labeled_inside_window(report, file_limit)
            if not flag:
                continue
            get_sources_for_report(repo, report, commit, full_save_path, file_limit)


def collect_sources_from_paths(repo, save_path, paths, file_limit=80):
    for path in paths:
        create_subfolder(save_path)
        get_sources_for_report(repo, path, save_path, file_limit=file_limit)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--intellij_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


def main():
    args = parse_args()

    intellij_path = args.intellij_path
    reports_path = args.reports_path
    files_limit = args.files_limit

    repo = Repo(intellij_path, odbt=db.GitDB)

    path_to_reports = os.path.join(reports_path, "labeled_reports")
    save_path = os.path.join(reports_path, "labeled_reports")
    create_subfolder(save_path)
    collect_sources_for_reports(repo, save_path, path_to_reports, files_limit)


if __name__ == "__main__":
    main()

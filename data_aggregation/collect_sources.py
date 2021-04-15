from git import Repo, db
import json
import os
from tqdm import tqdm
import re
import sys
from add_path_info import load_report, create_subfolder


def remove_unused_begin(code):
    clean_code = []
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


def get_sources_for_report(report, commit, full_save_path, file_limit=80):
    for i, frame in enumerate(report['frames'][:file_limit]):
        if i == file_limit:
            break
        if frame['path'] != '':
            diff_file = frame['path']
            code = get_file_by_commit(repo, commit+"~1", diff_file)
            f = open(os.path.join(full_save_path, frame['file_name']), 'w', encoding="utf-8")
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
            get_sources_for_report(report, commit, full_save_path, file_limit)


PATH_TO_INTELLIJ = os.path.join("..", "intellij-community")
PATH_TO_REPORTS = os.path.join("..", "intellij_fixed_201007")
FILES_LIMIT = 80
if __name__ == "__main__":
    path = PATH_TO_INTELLIJ
    repo = Repo(path, odbt=db.GitDB)

    files_limit = FILES_LIMIT
    if len(sys.argv) > 1:
        files_limit = sys.argv[3]
        PATH_TO_INTELLIJ = sys.argv[1]
        PATH_TO_REPORTS = sys.argv[2]
    path_to_reports = os.path.join(PATH_TO_REPORTS, "labeled_reports")
    save_path = os.path.join(PATH_TO_REPORTS, "labeled_reports")
    create_subfolder(save_path)
    collect_sources_for_reports(repo, save_path, path_to_reports, files_limit)

from git import Repo, db
import json
import os
from tqdm import tqdm
import re
import sys
from add_path_info import load_report
path = "C:\\Users\\lissrbay\\Desktop\\bugml\\intellij"


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


def create_subfolder(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def is_labeled_inside_window(report, file_limit):
    flag = 0
    for i, frame in enumerate(report['frames']):
        if i == file_limit:
            break
        if frame['file_name'] and frame['label']:
            flag = 1
    return flag


def get_sources_for_report(report, commit, full_save_path, file_limit):
    for i, frame in enumerate(report['frames']):
        if i == file_limit:
            break
        if frame['path'] != '':
            diff_file = frame['path']
            code = get_file_by_commit(repo, commit+"~1", diff_file)
            f = open(full_save_path + frame['file_name'], 'w', encoding="utf-8")
            f.write(code)
            f.close()


def collect_sources_for_reports(repo, save_path, path_to_reports, file_limit=80):
    for root, dirs, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            report = load_report(path_to_reports + "//" + file)
            commit = report['hash']
            if commit == "":
                continue
            full_save_path = save_path + "//" + str(report['id']) + "//"
            create_subfolder(full_save_path)

            flag = is_labeled_inside_window(report, file_limit)
            if not flag:
                continue
            get_sources_for_report(report, commit, full_save_path, file_limit)



if __name__ == "__main__":
    path = "//intellij"
    repo = Repo(path, odbt=db.GitDB)
    path_to_reports = "//labeled_reports"
    save_path = "//labeled_reports"
    if len(sys.argv) > 1:
        file_limit = sys.argv[1]
        collect_sources_for_reports(repo, save_path, path_to_reports, file_limit)
    else:
        collect_sources_for_reports(repo, save_path, path_to_reports)

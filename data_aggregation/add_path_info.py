from collections import defaultdict
import json
import os
from tqdm import tqdm
from git import Repo, db
import sys


def list_files_in_commit(commit, repo):
    configFiles = repo.git.execute(
        ['git', 'ls-tree', '-r', '--name-only', commit.hexsha]).split()
    java_files = defaultdict(list)
    for file in configFiles:
        if file.endswith('.java'):
            java_files[file.split('/')[-1]].append(file)

    return java_files


def get_method_path_and_name(frame):
    path = frame['method_name']
    path = path.split('$')[0]
    path = path.split('.')
    method = path.pop()
    path = "/".join(path)
    return path, method


def save_report(name, report):
    f = open(name, 'w')
    json.dump(report, f, indent=4)
    f.close()


def load_report(name):
    try:
        f = open(name, 'r')
        report = json.load(f)
        f.close()
    except Exception:
        return {}
    return report


def create_subfolder(path):
    try:
        os.mkdir(path)
    except Exception:
        # dir already exist
        pass 


def find_file_for_frame(frame, matching_files):
    frame['path'] = ""
    for file_path in matching_files:
        if frame['file_name'] in file_path:
            frame['path'] = file_path
            break
        if not 'path' in frame:
            frame['path'] = ""
    return frame


def add_paths_to_report(report, commit_files, file_limit=80):
    for i, frame in enumerate(report['frames']):
        if i == file_limit:
            break
        matching_files_for_frame = commit_files[frame['file_name']]
        frame = find_file_for_frame(frame, matching_files_for_frame)

    return report


def add_paths_to_all_reports(from_repo, path_to_reports, path_to_reports_save, file_limit=80):
    for root, _, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            path_to_file = os.path.join(path_to_reports, file)
            report = load_report(path_to_file)
            if report == {}:
                continue
            hash = report['hash']
            commit = from_repo.commit(hash + '~1')
            commit_files = list_files_in_commit(commit, from_repo)

            add_paths_to_report(report, commit_files, file_limit=80)
            save_report(os.path.join(path_to_reports_save, file), report)


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
    path_to_reports = os.path.join(PATH_TO_REPORTS, "reports")
    path_to_reports_save = os.path.join(PATH_TO_REPORTS, "labeled_reports")
    create_subfolder(path_to_reports_save)

    add_paths_to_all_reports(repo, path_to_reports, path_to_reports_save, files_limit)

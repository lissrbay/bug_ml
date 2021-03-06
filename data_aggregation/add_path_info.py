from collections import defaultdict
import json
import os
from tqdm import tqdm
from git import Repo, db

def list_files_in_commit(commit):
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
    f = open(name, 'r')
    report = json.load(f)
    f.close()
    return report

def find_file_for_frame(frame, matching_files):
    frame['path'] = ""
    for file_path in matching_files:
        if path in file_path:
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
        frame = find_file_for_frame(frame, matching_files)

    return report


def add_paths_to_all_reports(from_repo, path_to_reports, file_limit=80):
    for root, _, files in os.walk(path_to_reports):
        if not (root == path_to_reports):
            continue
        for file in tqdm(files):
            report = load_report(path_to_reports + "//" + file)

            hash = report['hash']
            commit = repo.commit(hash + '~1')
            commit_files = list_files_in_commit(commit)

            add_paths_to_report(report, commit_files, file_limit=80)
            save_report(path_to_reports + "//" + file, report)


if __name__ == "__main__":
    path = "//intellij"
    repo = Repo(path, odbt=db.GitDB)
    path_to_reports = "//labeled_reports"

    if len(sys.argv) > 1:
        files_limit = sys.argv[1]
        add_paths_to_all_reports(repo, path_to_reports, files_limit)
    else:
        add_paths_to_all_reports(repo, path_to_reports)
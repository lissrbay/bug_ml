import json
import os
import re
from argparse import ArgumentParser
from typing import Dict, Set, List, Tuple
from os.path import join
from tqdm import tqdm

from new.constants import REPO_CHANGED_METHODS_FILE, REPO_COMMIT_INFO_FILE
from new.data_aggregation.changes.get_java_methods import find_changed_methods, ChangedMethodSignature


def load_commits_log(data_dir: str) -> str:
    file_path = join(data_dir, REPO_COMMIT_INFO_FILE)
    with open(file_path, "r") as file:
        return file.read()


def get_issue_to_commit_hash(commits_log: str) -> Dict[str, str]:
    #  commits_log = load_commits_log(data_dir)

    # Parse commit hashes and start commit position.
    commit_pattern = re.compile("(?<=\ncommit )\w{40,40}")
    commits = []
    for commit_math in re.finditer(commit_pattern, commits_log):
        commit_hash, commit_start_pos = commit_math.group(0), commit_math.start()
        commits.append((commit_hash, commit_start_pos))
    commits.append(("", len(commits_log)))

    # Match issue to commit hash when the issue was fixed.
    issue_pattern = re.compile("(?<=EA-)\d+")
    issue_to_commit_hash = {}
    for ((commit_hash, commit_start_pos), (_, commit_end_pos)) in zip(commits, commits[1:]):
        commit_info = commits_log[commit_start_pos:commit_end_pos]
        issue_ids = re.findall(issue_pattern, commit_info)
        for issue_id in issue_ids:
            issue_to_commit_hash[issue_id] = str(commit_hash)

    return issue_to_commit_hash


def get_commit_changed_methods(repo_path: str, issue2commit: Dict[str, str]) -> Dict[str, Set[ChangedMethodSignature]]:
    return {
        commit: find_changed_methods(repo_path, (f"{commit}~1", commit)) for _, commit in tqdm(issue2commit.items())
    }


def parse_method_signature(changed_methods: Set[ChangedMethodSignature]) -> List[Tuple[str, str]]:
    methods = []
    for full_method_name, _ in changed_methods:
        filename = full_method_name.split(': ')[0]
        method = full_method_name.split(': ')[-1].split()[-1]
        methods.append((filename, method))
    return methods


def save_fixed_methods(data_dir: str, issue2commit: Dict[str, str],
                       commit_changed_methods: Dict[str, Set[ChangedMethodSignature]]):
    info = dict()
    for issue_id, commit in issue2commit.items():
        changed_methods = commit_changed_methods[commit]
        methods = []
        if changed_methods:
            methods = [{"path": method[0], "name": method[1]} for method in parse_method_signature(changed_methods)]
        info[issue_id] = {"hash": commit, "fixed_methods": methods}

    with open(os.path.join(data_dir, REPO_CHANGED_METHODS_FILE), 'w') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def get_changed_methods(repo_path: str, data_dir: str):
    commits_log = load_commits_log(data_dir)
    issue_to_commit_hash = get_issue_to_commit_hash(commits_log)
    commit_changed_methods = get_commit_changed_methods(repo_path, issue2commit)
    save_fixed_methods(data_dir, issue2commit, commit_changed_methods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    get_changed_methods(args.repo_path, args.data_dir)

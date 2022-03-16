import json
import os
import re
from argparse import ArgumentParser
from typing import Dict, Set, List, Tuple

from tqdm import tqdm

from new.constants import REPO_CHANGED_METHODS_FILE, REPO_COMMIT_INFO_FILE
from new.data_aggregation.changes.get_java_methods import find_changed_methods, ChangedMethodSignature


def get_commits_and_issues(data_dir: str) -> Dict[str, str]:
    with open(os.path.join(data_dir, REPO_COMMIT_INFO_FILE), "r") as f:
        commits_info = f.read()
    pattern_commit = re.compile("(?<=\ncommit )\w{40,40}")
    pattern_issue = re.compile("(?<=EA-)\d+")
    commits = [(commit.group(0), commit.start()) for commit in re.finditer(pattern_commit, commits_info)]
    commits.append(("", len(commits_info)))
    issue2hash = {}
    for i in range(len(commits) - 1):
        commit_text = commits_info[commits[i][1]: commits[i + 1][1]]
        issue_ids = re.findall(pattern_issue, commit_text)
        for issue_id in issue_ids:
            issue2hash[issue_id] = str(commits[i][0])
    return issue2hash


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


def get_all_changed_methods(repo_path: str, data_dir: str):
    issue2commit = get_commits_and_issues(data_dir)
    commit_changed_methods = get_commit_changed_methods(repo_path, issue2commit)
    save_fixed_methods(data_dir, issue2commit, commit_changed_methods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    get_all_changed_methods(args.repo_path, args.data_dir)

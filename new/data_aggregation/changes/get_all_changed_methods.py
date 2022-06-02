import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, Set

from tqdm import tqdm
import pandas as pd
from new.constants import REPO_CHANGED_METHODS_FILE, REPO_COMMIT_INFO_FILE, ISSUE_REPORTS_MAPPING_FILE
from new.data_aggregation.changes.get_java_methods import find_changed_methods, MethodSignature
from new.data_aggregation.utils import parse_method_signature


def get_commits_and_issues(data_dir: str, reports_path: str) -> Dict[str, str]:
    with open(os.path.join(data_dir, REPO_COMMIT_INFO_FILE), "r") as f:
        commits_info = f.read()

    issues_reports = set(pd.read_csv(os.path.join(reports_path, ISSUE_REPORTS_MAPPING_FILE)).issue_id.values)

    pattern_commit = re.compile("(?<=\ncommit )\w{40,40}")
    pattern_issue = re.compile("(?<= EA-)\d+")
    commits = [(commit.group(0), commit.start()) for commit in re.finditer(pattern_commit, commits_info)]
    commits.append(("", len(commits_info)))
    issue2hash = defaultdict(list)
    for i in range(len(commits) - 1):
        commit_text = commits_info[commits[i][1]: commits[i + 1][1]]
        issue_ids = re.findall(pattern_issue, commit_text)
        for issue_id in issue_ids:
            if int(issue_id) in issues_reports:
                issue2hash[issue_id].append(commits[i][0])


    return issue2hash


def get_commit_changed_methods(repo_path: str, issue2commit: Dict[str, str]) -> Dict[str, Set[MethodSignature]]:
    changed_methods = {}
    for _, commits in tqdm(list(issue2commit.items())):
        for commit in commits:
            changed_methods[commit] = find_changed_methods(repo_path, (f"{commit}~1", commit))
    return changed_methods


def save_fixed_methods(data_dir: str, issue2commit: Dict[str, str],
                       commit_changed_methods: Dict[str, Set[MethodSignature]]):
    info = dict()
    for issue_id, commits in issue2commit.items():
        methods = []
        for commit in commits:
            if commit in commit_changed_methods:
                changed_methods = commit_changed_methods[commit]
                if changed_methods:
                    print(changed_methods)
                    methods.extend(
                        [{"path": method[0], "name": method[1]} for method in parse_method_signature(changed_methods)])
        info[issue_id] = {"hash": commits[0], "fixed_methods": methods}

    with open(os.path.join(data_dir, REPO_CHANGED_METHODS_FILE), 'w') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def get_all_changed_methods(repo_path: str, reports_path: str, data_dir: str):
    issue2commit = get_commits_and_issues(data_dir, reports_path)
    commit_changed_methods = get_commit_changed_methods(repo_path, issue2commit)
    save_fixed_methods(data_dir, issue2commit, commit_changed_methods)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    get_all_changed_methods(args.repo_path, args.data_dir)

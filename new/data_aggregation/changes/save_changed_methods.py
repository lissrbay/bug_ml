import json
import re
from argparse import ArgumentParser
from os.path import join
from typing import Dict, Set, List, Tuple, Any

from git import Repo, db
from tqdm import tqdm

from new.constants import REPO_CHANGED_METHODS_FILE, REPO_COMMIT_INFO_FILE
from new.data_aggregation.changes.get_java_methods import find_changed_methods, MethodSignature


def load_commits_log(data_dir: str) -> str:
    file_path = join(data_dir, REPO_COMMIT_INFO_FILE)
    with open(file_path, "r") as file:
        return file.read()


def save_methods(issue_to_methods: Dict[str, Dict[str, Any]], data_dir: str):
    file_path = join(data_dir, REPO_CHANGED_METHODS_FILE)
    with open(file_path, "w") as file:
        json.dump(issue_to_methods, file, indent=4, ensure_ascii=False)


def get_issue_to_commit(commits_log: str) -> Dict[str, str]:
    """
    Match issue to its fixed commit hash.
    """

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


def get_commit_changed_methods(repo: Repo, issue_to_commit: Dict[str, str]) -> Dict[str, Set[MethodSignature]]:
    commit_to_methods = {}

    for commit_hash in tqdm(issue_to_commit.values()):
        commit_to_methods[commit_hash] = find_changed_methods(repo, (f"{commit_hash}~1", commit_hash))

    return commit_to_methods


def parse_method_signature(changed_methods: Set[MethodSignature]) -> List[Tuple[str, str]]:
    methods = []

    for full_method_name, _ in changed_methods:
        file_name = full_method_name.split(': ')[0]
        method_name = full_method_name.split(': ')[-1].split()[-1]
        methods.append((file_name, method_name))

    return methods


def get_fixed_methods(issue_to_commit: Dict[str, str],
                      commit_to_changed_methods: Dict[str, Set[MethodSignature]]) -> Dict[str, Dict[str, Any]]:
    issue_to_methods = {}
    for issue_id, commit in issue_to_commit.items():
        changed_methods = commit_to_changed_methods[commit]
        methods = []

        if changed_methods:
            methods = []
            for file_name, method_name in parse_method_signature(changed_methods):
                method_dict = {"path": file_name, "name": method_name}
                methods.append(method_dict)

        issue_to_methods[issue_id] = {"hash": commit, "fixed_methods": methods}

    return issue_to_methods


def save_changed_methods(repo_path: str, data_dir: str):
    """
    Gets changed methods and saved to data directory.
    """
    commits_log = load_commits_log(data_dir)
    repo = Repo(repo_path, odbt=db.GitDB)

    issue_to_commit = get_issue_to_commit(commits_log)
    commit_changed_methods = get_commit_changed_methods(repo, issue_to_commit)
    issue_to_methods = get_fixed_methods(issue_to_commit, commit_changed_methods)
    save_methods(issue_to_methods, data_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    save_changed_methods(args.repo_path, args.data_dir)

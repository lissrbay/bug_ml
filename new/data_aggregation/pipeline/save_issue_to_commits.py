import argparse
import json
import re
from collections import defaultdict
from os.path import join
from typing import Dict, List

from new.constants import ISSUE_TO_COMMITS_FILE, FIX_COMMITS_FILE


def save(issue_to_commits: Dict[int, List[str]], data_dir: str):
    save_path = join(data_dir, ISSUE_TO_COMMITS_FILE)
    with open(save_path, "w") as file:
        json.dump(issue_to_commits, file, indent=4)


def load_commit_log(data_dir: str) -> str:
    load_path = join(data_dir, FIX_COMMITS_FILE)
    with open(load_path, "r") as file:
        return file.read()


def get_issue_to_commits(commit_log: str) -> Dict[int, List[str]]:
    # Parse commit hashes and start commit position.
    commit_pattern = re.compile("(?<=\ncommit )\w{40,40}")
    commits = []
    for commit_math in re.finditer(commit_pattern, commit_log):
        commit_hash, commit_start_pos = commit_math.group(0), commit_math.start()
        commits.append((commit_hash, commit_start_pos))
    commits.append(("", len(commit_log)))

    # Match issue to commit hash when the issue was fixed.
    issue_pattern = re.compile("(?<=EA-)\d+")
    issue_to_commits = defaultdict(list)
    for ((commit_hash, commit_start_pos), (_, commit_end_pos)) in zip(commits, commits[1:]):
        commit_info = commit_log[commit_start_pos:commit_end_pos]
        issue_ids = re.findall(issue_pattern, commit_info)
        for issue_id in map(int, issue_ids):
            issue_to_commits[issue_id].append(str(commit_hash))

    return issue_to_commits


def save_issue_to_commits(data_dir: str):
    """
    Parses issue to commit hash from commit log and saves to data directory.
    """
    print(f"Parsing issue to commit hash")
    commit_log = load_commit_log(data_dir)
    issue_to_commits = get_issue_to_commits(commit_log)
    save(issue_to_commits, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    save_issue_to_commits(args.data_dir)

import sys
import argparse
import json
import os
import re
from collections import defaultdict

from tqdm import tqdm

from get_java_methods import ChangedMethodsFinder
cmf = ChangedMethodsFinder()

def get_changed_methods_from_commits(next_commit, path):
    changed_methods = cmf.find_changed_methods(path, [next_commit + '~1', next_commit])
    return changed_methods


def get_commits_and_issues():
    path_to_fix_commits = os.path.join(".", "data", "commit_fix_hashes.txt")

    f = open(path_to_fix_commits, "r")
    commits_info = "".join(f.readlines())
    pattern_commit = re.compile("(?<=\ncommit )\w{40,40}")
    pattern_issue = re.compile("(?<=EA-)\d+")
    issues = []
    commits = [(commit.group(0), commit.start()) for commit in re.finditer(pattern_commit, commits_info)]
    commits.append(("", len(commits_info)))
    hashes = []
    for i in range(len(commits) - 1):
        commit_text = commits_info[commits[i][1]: commits[i + 1][1]]
        issue_ids = re.findall(pattern_issue, commit_text)
        for issue_id in issue_ids:
            issues.append(issue_id)
            hashes.append(str(commits[i][0]))
    return hashes, issues


def collect_all_changed_methods(fix_commits_hashes, path):
    changed_methods = list()
    for commit in tqdm(fix_commits_hashes):
        changed_methods.append(get_changed_methods_from_commits(commit, path))
    return changed_methods


def parse_method_signature(cms):
    methods = []
    for method_signature in cms:
        full_method_name = method_signature[0]
        filename = full_method_name.split(': ')[0]
        method = full_method_name.split(': ')[-1].split()[-1]
        methods.append((filename, method))
    return methods


def save_results(fix_commit_hashes, fix_issues, changed_methods):
    info = dict()
    for i in range(len(changed_methods)):
        issue = fix_issues[i]
        if changed_methods[i]:
            cms = list(changed_methods[i])
            methods = parse_method_signature(cms)
            info[issue] = {"hash": fix_commit_hashes[i], "fixed_methods": methods}
        else:
            info[issue] = {"hash": fix_commit_hashes[i], "fixed_methods": []}

    f = open(os.path.join(".", "data", "fixed_methods.txt"), 'w')
    json.dump(info, f, indent=4)
    f.close()


def main(intellij_path):
    fix_commits_hashes, fix_issues = get_commits_and_issues()
    changed_methods = collect_all_changed_methods(fix_commits_hashes, intellij_path)
    save_results(fix_commits_hashes, fix_issues, changed_methods)


if __name__ == "__main__":
    main('/Users/e.poslovskaya/bug_ml_copy_2/bug_ml_copy_2/intellij-community')

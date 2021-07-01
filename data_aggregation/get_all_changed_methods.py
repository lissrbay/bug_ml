from get_java_methods import ChangedMethodsFinder
from tqdm import tqdm
import re
from collections import defaultdict
import json
import os
import sys
PATH_TO_INTELLIJ = os.path.join("..", "intellij-community")

def get_changed_methods_from_commits(next_commit, path=PATH_TO_INTELLIJ):
    cmf = ChangedMethodsFinder()
    
    changed_methods = cmf.find_changed_methods(path, [next_commit + '~1', next_commit])
    #print(changed_methods)
    return changed_methods



def get_commits_and_issues(path):
    f = open(path, "r")
    commits_info  = "".join(f.readlines())
    pattern_commit = re.compile("(?<=\ncommit )\w{40,40}")
    pattern_issue = re.compile("(?<=EA-)\d+")
    issues = defaultdict(list)
    commits = [(commit.group(0), commit.start()) for commit in re.finditer(pattern_commit, commits_info)]
    commits.append(("", len(commits_info)))

    for i in range(len(commits)-1):
        commit_text = commits_info[commits[i][1]: commits[i+1][1]]
        issue_id = re.search(pattern_issue, commit_text)
        issues[str(commits[i][0])] = issue_id.group(0)
    return list(issues.keys()), issues


def collect_all_changed_methods(fix_commits_hashes):
    changed_methods = list()
    for commit in tqdm(fix_commits_hashes):
        changed_methods.append(get_changed_methods_from_commits(commit))
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
        if changed_methods[i]:
            cms = list(changed_methods[i])
            methods = parse_method_signature(cms)
            issue = fix_issues[fix_commit_hashes[i]]

            info[issue] = {"hash" : fix_commit_hashes[i], "fixed_methods" : methods}
        else:
            issue = fix_issues[fix_commit_hashes[i]]
            print(str(issue))
            info[issue] = {"hash" : fix_commit_hashes[i], "fixed_methods" : []}

    f = open("fixed_methods.txt", 'w')
    json.dump(info, f, indent=4)
    f.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PATH_TO_INTELLIJ = sys.argv[1]
    path_to_fix_commits = os.path.join(".", "commit_fix_hashes.txt")
    fix_commits_hashes, fix_issues = get_commits_and_issues(path_to_fix_commits)
    changed_methods = collect_all_changed_methods(fix_commits_hashes)
    save_results(fix_commits_hashes, fix_issues, changed_methods)
    
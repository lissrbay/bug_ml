from get_java_methods import ChangedMethodsFinder
from tqdm import tqdm
import re
from collections import defaultdict
import json


def get_changed_methods_from_commits(next_commit):
    path = "C:\\Users\\lissrbay\\Desktop\\bugml\\intellij"
    cmf = ChangedMethodsFinder()

    changed_methods = cmf.find_changed_methods(path, [next_commit + '~1', next_commit])
    return changed_methods



def get_commits_and_issues(path):
    f = open(path, "r")
    pattern_class = re.compile("(?<=commit )[\w]+")
    pattern_issue = re.compile("(?:^|\s)EA-[\d]+")
    commits = list()
    issues = defaultdict(list)

    for line in f.readlines():
        if len(line) == 48:
            commits_founded = re.findall(pattern_class, line)
            if commits_founded and len(commits_founded[-1]) == 40:
                commits.append(commits_founded[-1])
            continue

        issues_founded = re.findall(pattern_issue, line)
        for issue in issues_founded:
            issues[commits[-1]].append(issue.strip()[3:])
    return commits, issues


def collect_all_changed_methods(fix_commits_hashes, fix_issues):
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
            for issue in fix_issues[fix_commits_hashes[i]]:
                info[issue] = {"hash" : fix_commits_hashes[i], "fixed_methods" : methods}
        else:
            for issue in fix_issues[fix_commits_hashes[i]]:
                info[issue] = {"hash" : fix_commits_hashes[i], "fixed_methods" : []}

    f = open("fixed_methods.txt", 'w')
    json.dump(info, f, indent=4)
    f.close()


if __name__ == "__main__":
    path_to_fix_commits = "./commit_fix_hashes.txt"
    fix_commits_hashes, fix_issues = get_commits_and_issues(path_to_fix_commits)
    changed_methods = collect_all_changed_methods(fix_commits_hashes, fix_issues)
    save_results(fix_commit_hashes, fix_issues, changed_methods)
    
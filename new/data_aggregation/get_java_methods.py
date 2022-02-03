# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py "<path or nothing>"

import re
from typing import Tuple, List, Optional, Set

from git import Repo, db

from new.constants import MAX_DIFF_FILES
from new.data_aggregation.parser_java_kotlin import Parser, AST


def is_match_lang_ext(filename: str):
    file_extension = {'java': '.*.java', 'kotlin': '.*.kt'}
    return re.match(file_extension['java'], filename) or re.match(file_extension['kotlin'], filename)


def collect_modified_files_last_two_commits(repo: Repo, commits: Tuple[str, str]) -> List[str]:
    commit_dev = repo.commit(commits[0])
    commit_origin_dev = repo.commit(commits[1])
    diff_index = commit_origin_dev.diff(commit_dev)
    diff_files: List[str] = []
    for diff_item in diff_index.iter_change_type('M'):
        diff_files.append(diff_item.b_path)
        if len(diff_files) > MAX_DIFF_FILES:
            return []
    diff_files = [f for f in diff_files if is_match_lang_ext(f) and not re.search('auto_generated', f)]

    return diff_files


def remove_tabs_and_comments(code: str) -> str:
    lines = code.split('\n')
    code = '\n'.join(filter(lambda x: not (x.strip()[:2] == '//'), lines))
    code = re.sub(' +', ' ', code)
    return re.sub('\t+', '', code)


def code_fragment(bounds: Tuple[int, int], code: str):
    if not bounds:
        return ''
    if bounds[1] <= bounds[0]:
        return ''
    return ''.join(code)[bounds[0]: bounds[1]]


def compare_ast(ast_a: AST, code_a: str, ast_b: AST, code_b: str) -> Set[Tuple[str, str]]:
    methods_info_a = dict(ast_a.get_method_names_and_bounds())
    methods_info_b = dict(ast_b.get_method_names_and_bounds())
    all_methods = list(methods_info_a.keys()) + list(methods_info_b.keys())
    changed_methods = set()
    for method in all_methods:
        if method in methods_info_a and method in methods_info_b:
            method_code_a = code_fragment(methods_info_a[method][0], code_a)
            method_code_b = code_fragment(methods_info_b[method][0], code_b)

            if method_code_a != method_code_b:
                changed_methods.add((method, methods_info_a[method][1]))
        if method in methods_info_a and not (method in methods_info_b):
            changed_methods.add((method, methods_info_a[method][1]))
    return changed_methods


def get_code(repo: Repo, diff_file: str, commit: str) -> Optional[str]:
    try:
        code = repo.git.show('{}:{}'.format(commit, diff_file))
        return remove_tabs_and_comments(code)
    except Exception:
        return None


def find_changed_methods_by_language(repo: Repo, language: str, diff_files: List[str],
                                     commits: Tuple[str, str] = None) -> Set[Tuple[str, str]]:
    commits = commits or ("HEAD", "HEAD~1")
    trees_a, trees_b = dict(), dict()
    codes_a, codes_b = dict(), dict()
    all_changed_methods = set()
    for diff_file in diff_files:
        codes_a[diff_file] = get_code(repo, diff_file, commits[0])  # TODO: space before code
        codes_b[diff_file] = get_code(repo, diff_file, commits[1])
        trees_a[diff_file] = Parser(language).parse(codes_a[diff_file], diff_file)  # TODO: write tests on it, maybe works wrong
        trees_b[diff_file] = Parser(language).parse(codes_b[diff_file], diff_file)
        all_changed_methods |= compare_ast(
            trees_a[diff_file], codes_a[diff_file], trees_b[diff_file], codes_b[diff_file]
        )
    return all_changed_methods


def find_changed_methods(repo_path: str, commits: Tuple[str, str] = None) -> Set[Tuple[str, str]]:
    commits = commits or ("HEAD", "HEAD~1")
    try:
        repo = Repo(repo_path, odbt=db.GitDB)
        diff_files = collect_modified_files_last_two_commits(repo, commits)
        java_changed_methods = find_changed_methods_by_language(repo, 'java', diff_files, commits)
        kotlin_changed_methods = find_changed_methods_by_language(repo, 'kotlin', diff_files, commits)
        return java_changed_methods.union(kotlin_changed_methods)
    except Exception as e:
        print("Check path to repository. Maybe, you should write path in double quotes\"\"")
        raise e

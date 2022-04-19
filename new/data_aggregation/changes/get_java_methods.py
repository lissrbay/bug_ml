# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py "<path or nothing>"

import re
from dataclasses import dataclass
from typing import Tuple, List, Optional, Set

import attr
from git import Repo

from new.constants import MAX_DIFF_FILES
from new.data_aggregation.changes.parser_java_kotlin import Parser, AST
import attrs


@dataclass(frozen=True)
class MethodSignature:
    name: Optional[str] = None
    type: Optional[str] = None


def is_lang_match(file_name: str) -> bool:
    patterns = [".*.java", ".*.kt"]

    for pattern in patterns:
        if re.match(pattern, file_name):
            return True

    return False


def is_autogen(file_name: str) -> bool:
    return "auto_generated" in file_name


def collect_modified_files_last_two_commits(repo: Repo, commits: Tuple[str, str]) -> List[str]:
    commit_dev = repo.commit(commits[0])
    commit_origin_dev = repo.commit(commits[1])
    diff_index = commit_origin_dev.diff(commit_dev)
    diff_files: List[str] = []
    for diff_item in diff_index.iter_change_type('M'):
        diff_files.append(diff_item.b_path)
        if len(diff_files) > MAX_DIFF_FILES:
            return []

    diff_files = [f for f in diff_files if is_lang_match(f) and not is_autogen(f)]

    return diff_files


def remove_tabs_and_comments(source_code: str) -> str:
    lines = source_code.split("\n")
    lines = filter(lambda line: not line.strip().startswith("//"), lines)  # Remove comments
    source_code = "\n".join(lines)
    source_code = re.sub(" +", " ", source_code)
    source_code = re.sub("\t+", "", source_code)
    return source_code


def cut_part(code: str, bounds: Optional[Tuple[int, int]]) -> str:
    if bounds is None:
        return ""

    start_pos, end_pos = bounds
    if end_pos <= start_pos:
        return ""

    part = "".join(code)[start_pos:end_pos]
    return part


def compare_ast(ast_a: AST, code_a: str, ast_b: AST, code_b: str) -> Set[MethodSignature]:
    methods_info_a = set(ast_a.get_method_names_and_bounds())
    methods_info_b = set(ast_b.get_method_names_and_bounds())
    all_methods = methods_info_a | methods_info_b
    changed_methods = set()
    for method in all_methods:
        if method in methods_info_a and method in methods_info_b:
            method_code_a = cut_part(code_a, method.bounds)
            method_code_b = cut_part(code_b, method.bounds)

            if method_code_a != method_code_b:
                changed_methods.add((method, method.type))
        if method in methods_info_a and not (method in methods_info_b):
            changed_methods.add(MethodSignature(method.name, method.type))
    return changed_methods


def get_code(repo: Repo, diff_file: str, commit: str) -> Optional[str]:
    try:
        code = repo.git.show(f"{commit}:{diff_file}")
        return remove_tabs_and_comments(code)
    except Exception as e:
        pass


def find_changed_methods_by_language(repo: Repo, language: str, diff_files: List[str],
                                     commits: Tuple[str, str] = None) -> Set[Tuple[str, str]]:
    commits = commits or ("HEAD", "HEAD~1")
    trees_a, trees_b = dict(), dict()
    codes_a, codes_b = dict(), dict()
    all_changed_methods = set()
    for diff_file in diff_files:
        codes_a[diff_file] = get_code(repo, diff_file, commits[0])  # TODO: space before code
        codes_b[diff_file] = get_code(repo, diff_file, commits[1])
        trees_a[diff_file] = Parser(language).parse(codes_a[diff_file],
                                                    diff_file)  # TODO: write tests on it, maybe works wrong
        trees_b[diff_file] = Parser(language).parse(codes_b[diff_file], diff_file)
        all_changed_methods |= compare_ast(
            trees_a[diff_file], codes_a[diff_file], trees_b[diff_file], codes_b[diff_file]
        )
    return all_changed_methods


def find_changed_methods(repo: Repo, commits: Tuple[str, str] = None) -> Set[Tuple[str, str]]:
    commits = commits or ("HEAD", "HEAD~1")
    try:
        diff_files = collect_modified_files_last_two_commits(repo, commits)
        java_changed_methods = find_changed_methods_by_language(repo, 'java', diff_files, commits)
        kotlin_changed_methods = find_changed_methods_by_language(repo, 'kotlin', diff_files, commits)
        changed_methods = java_changed_methods | kotlin_changed_methods
        return changed_methods
    except Exception as e:
        # print("Check path to repository. Maybe, you should write path in double quotes\"\"")
        print(f"Exception with {e}")
        raise e

# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py "<path or nothing>"

from git import Repo, db
import os.path
import re
import sys
import glob
from .parser_java_kotlin import Parser
from pathlib import Path
from tqdm import tqdm

class ChangedMethodsFinder:
    file_extension = {'java': '.*.java', 'kotlin':'.*.kt'}

    def __init__(self, path='.'):
        self.repo = None
        self.path = path
        self.code_a = ''
        self.code_b = ''


    def collect_code_from_commit(self, diff_file, commit_step):
        try:
            return self.repo.git.show('{}:{}'.format(commit_step, diff_file)).split('\n')
        except Exception:
            return['error']

    def is_match_lang_ext(self, filename):
        return (re.match(self.file_extension['java'], filename) or re.match(self.file_extension['kotlin'], filename))


    def collect_modified_files_last_two_commits(self,  commits = ["HEAD", "HEAD~1"]):
        commit_dev = self.repo.commit(commits[0])
        commit_origin_dev = self.repo.commit(commits[1])
        diff_index = commit_origin_dev.diff(commit_dev)
        diff_files = []
        for diff_item in diff_index.iter_change_type('M'):
            diff_files.append(diff_item.b_path)
            if len(diff_files) > 20:
                return []
        diff_files = [f for f in diff_files if self.is_match_lang_ext(f) and not re.search('auto_generated', f)]

        return diff_files


    def remove_tabs(self, code):
        code = list(filter(lambda x: not (x.strip()[:2] == '//'), code))
        code = '\n'.join(code)
        code = re.sub(' +', ' ', code)
        return re.sub('\t+', '', code)


    def open_repo(self, path='.'):
        try:
            self.repo = Repo(path, odbt=db.GitDB)
        except Exception:
            print("Check path to repository. Maybe, you should write path in double quotes\"\"")


    def code_fragment(self, bounds, code):
        if not bounds:
            return ''
        if bounds[1]<= bounds[0]:
            return ''
        return ''.join(code)[bounds[0]: bounds[1]]


    def get_method_info(self, ast):
        methods_info = ast.get_method_names_and_bounds()
        methods_info = dict(methods_info)
        return methods_info


    def compare_ast(self, ast_a, ast_b, diff_file):
        methods_info_a = self.get_method_info(ast_a)
        methods_info_b = self.get_method_info(ast_b)
        all_methods = list(methods_info_a.keys()) + list(methods_info_b.keys())
        changed_methods = set()
        for method in all_methods:
            if method in methods_info_a and method in methods_info_b:
                method_code_a = self.code_fragment(methods_info_a[method][0], self.codes_a[diff_file])
                method_code_b = self.code_fragment(methods_info_b[method][0], self.codes_b[diff_file])

                if method_code_a != method_code_b:
                    changed_methods.add((method, methods_info_a[method][1]))
            if method in methods_info_a and not (method in methods_info_b):
                changed_methods.add((method, methods_info_a[method][1]))
        return changed_methods


    def get_code(self, diff_file, commit):
        code = self.collect_code_from_commit(diff_file, commit)
        code = self.remove_tabs(code)
        return code


    def construct_ast(self, code, language='java', diff_file=''):
        parser = Parser(language)
        ast = parser.parse(code, diff_file)
        return ast


    def find_changed_methods_by_language(self, language='java', diff_files=[], commits=["HEAD", "HEAD~1"]):
        self.trees_a, self.trees_b = dict(), dict()
        self.codes_a, self.codes_b = dict(), dict()
        all_changed_methods = set()
        for diff_file in diff_files:
            self.codes_a[diff_file] = self.get_code(diff_file, commits[0])
            self.codes_b[diff_file] = self.get_code(diff_file, commits[1])
            self.trees_a[diff_file] = self.construct_ast(self.codes_a[diff_file], language, diff_file)
            self.trees_b[diff_file] = self.construct_ast(self.codes_b[diff_file], language, diff_file)
            all_changed_methods = all_changed_methods.union(self.compare_ast(self.trees_a[diff_file],
                                                            self.trees_b[diff_file], diff_file))
        return all_changed_methods


    def find_changed_methods(self, path='.', commits = ["HEAD", "HEAD~1"]):
        self.open_repo(path)
        diff_files = self.collect_modified_files_last_two_commits(commits)
        java_changed_methods = self.find_changed_methods_by_language('java', diff_files, commits)
        kotlin_changed_methods = self.find_changed_methods_by_language('kotlin', diff_files, commits)
        return java_changed_methods.union(kotlin_changed_methods)


if __name__ == "__main__":
    path = '.'
    if len(sys.argv) > 1:
            path = sys.argv[1]
    cmf = ChangedMethodsFinder()
    commits = ['ecdd37cc44f9beb6870c78c3432b1fddcdab8292~1','ecdd37cc44f9beb6870c78c3432b1fddcdab8292']
    print(cmf.find_changed_methods(path, commits))

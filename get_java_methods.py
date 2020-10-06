# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py "<path or nothing>"

from git import Repo, db
import os.path
import re
import sys
import glob
from parser_java_kotlin import Parser
from pathlib import Path
from tqdm import tqdm

class ChangedMethodsFinder:
    file_extension = {'java': '.*.java', 'kotlin':'.*.kt'}

    def __init__(self, path='.'):
        self.repo = None
        self.path = path
        self.code_a = ''
        self.code_b = ''


    def collect_code_from_commit(self, diff_files, commit_step, extension='.*.java'):
        code = []
        for diff_file in tqdm(diff_files):
            code.extend(self.repo.git.show('{}:{}'.format(commit_step, diff_file)).split('\n'))

        return code


    def collect_code_last_two_commits(self, extension='.*.java', commits = ["HEAD", "HEAD~1"]):
        commit_dev = self.repo.commit(commits[0])
        commit_origin_dev = self.repo.commit(commits[1])
        diff_index = commit_origin_dev.diff(commit_dev)
        diff_files = []
        for diff_item in diff_index.iter_change_type('M'):
            diff_files.append(diff_item.a_path)
        self.code_a = self.collect_code_from_commit(diff_files, commits[0], extension)
        self.code_b = self.collect_code_from_commit(diff_files, commits[1], extension)
        print("collected")


    def remove_tabs(self, code):
        code = '\n'.join(code)
        code = re.sub(' +', ' ', code)
        return re.sub('\t+', '', code)


    def parse_code(self, code):
        classes = {m.start(0):m.group(0) for m in re.finditer(self.pattern_class, code)}
        methods = [(m.start(0), m.end(0), m.group(0))  for m in re.finditer(self.pattern_method_all, code)]
        methods_names = re.findall(self.pattern_method_name, code)
        return classes, methods, methods_names


    def open_repo(self, path='.'):
        try:
            self.repo = Repo(path, odbt=db.GitDB)
        except Exception:
            print("Check path to repository. Maybe, you should write path in double quotes\"\"")


    def code_fragment(self, bounds, code):
        if not bounds:
            return ''

        return ''.join(code)[bounds[0]: bounds[1]]


    def get_method_info(self, ast):
        methods_info = ast.get_method_names_and_bounds()
        methods_info = dict(methods_info)

        return methods_info


    def compare_ast(self, ast_a, ast_b):
        methods_info_a = self.get_method_info(ast_a)
        methods_info_b = self.get_method_info(ast_b)
        all_methods = list(methods_info_a.keys()) + list(methods_info_b.keys())
        changed_methods = set()

        for method in all_methods:
            if method in methods_info_a and method in methods_info_b:
                method_code_a = self.code_fragment(methods_info_a[method][0], self.code_a)
                method_code_b = self.code_fragment(methods_info_b[method][0], self.code_b)
                if method_code_a != method_code_b:
                    changed_methods.add((method, methods_info_a[method][1]))

        return changed_methods


    def find_changed_methods_by_language(self, language = 'java', commits = ["HEAD", "HEAD~1"]):
        extension = self.file_extension[language]
        self.collect_code_last_two_commits(extension, commits)
        parser = Parser(language)
        self.code_a = self.remove_tabs(self.code_a)
        ast_a = parser.parse(self.code_a)
        parser = Parser(language)
        self.code_b = self.remove_tabs(self.code_b)
        ast_b = parser.parse(self.code_b)
        return self.compare_ast(ast_a, ast_b)
        
    def find_changed_methods(self, path='.', commits = ["HEAD", "HEAD~1"]):
        self.open_repo(path)
        java_changed_methods = self.find_changed_methods_by_language('java', commits)
        kotlin_changed_methods = self.find_changed_methods_by_language('kotlin', commits)
        return java_changed_methods.union(kotlin_changed_methods)


if __name__ == "__main__":
    path = '.'
    if len(sys.argv) > 1:
            path = sys.argv[1]
            path = path.replace("\\", "/")
    cmf = ChangedMethodsFinder()
    commits = ['8b5304787d3a869397a45ab9f355b29d31aabd6c','c24ccb312382238ee9fc3198c7851b16d19c9563']
    print(cmf.find_changed_methods(path, commits))
    print(cmf.find_changed_methods(path))


# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py "<path or nothing>"
from git import Repo, db
import os.path
import re
import sys
from java_parser import JavaParser


class ChangedMethodsFinder:
    pattern_method_all = re.compile('(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*\{?[^\}]*\}?')
    pattern_method_name = re.compile('(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*?')
    pattern_class = re.compile("(?:public|protected|private|static)\s+(?:class|interface)\s+\w+\s*")


    def __init__(self, path='.'):
        self.repo = None
        self.path = path
        self.changed_methods = []
        self.code_a = ''
        self.code_b = ''


    def collect_code_from_commit(self, diff_files, commit_step):
        code = []
        for diff_file in diff_files.split('\n'):
            if re.match('.*.java', diff_file):
                for commit, code_lines in self.repo.blame(commit_step, diff_file):
                        code.extend(code_lines)
        return code


    def collect_code_last_two_commits(self):
        diff_files = self.repo.git.diff('HEAD~1..HEAD', name_only=True)
        self.code_a = self.collect_code_from_commit(diff_files, "HEAD")
        self.code_b = self.collect_code_from_commit(diff_files, "HEAD~1")


    def remove_tabs(self, code):
        return re.sub(r"[\t]*", '', '\n'.join(code))


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
                method_code_a = self.code_fragment(methods_info_a[method], self.code_a)
                method_code_b = self.code_fragment(methods_info_b[method], self.code_b)
                if method_code_a != method_code_b:
                    changed_methods.add(method + " - changed")

        return changed_methods


    def find_changed_methods(self, path='.'):
        self.open_repo(path)
        self.collect_code_last_two_commits()
        jp = JavaParser()
        ast_a = jp.parse(self.code_a)
        ast_b = jp.parse(self.code_b)

        return self.compare_ast(ast_a, ast_b)


if __name__ == "__main__":
    path = '.'
    if len(sys.argv) > 1:
            path = sys.argv[1]
            path = path.replace("\\", "/")
    cmf = ChangedMethodsFinder(path)
    print(cmf.find_changed_methods(path))


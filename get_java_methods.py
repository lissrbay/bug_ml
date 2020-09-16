# This script finds changed methods between two last commits in local repository.
# To run the script, type python get_java_methods.py <path or nothing>

from git import Repo, db
import os.path
import re
import sys


class ChangedMethodsFinder:
    pattern_method_all = re.compile('(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*\{?[^\}]*\}?')
    pattern_method_name = re.compile('(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*?')
    pattern_class = re.compile("(?:public|protected|private|static)\s+(?:class|interface)\s+\w+\s*")


    def __init__(self, path='.'):
        self.repo = None
        self.path = path
        self.changed_methods = []


    def collect_code_from_commit(self, diff_files, commit_step):
        code = []
        for diff_file in diff_files.split('\n'):
            if re.match('.*.java', diff_file):
                for commit, code_lines in self.repo.blame(commit_step, diff_file):
                        code.extend(code_lines)
        return code


    def collect_code_last_two_commits(self):
        diff_files = self.repo.git.diff('HEAD~1..HEAD', name_only=True)
        code_a = self.collect_code_from_commit(diff_files, "HEAD")
        code_b = self.collect_code_from_commit(diff_files, "HEAD~1")
        return code_a, code_b


    def remove_tabs(self, code):
        return re.sub(r"[\t]*", '', '\n'.join(code))


    def parse_code(self, code):
        classes = {m.start(0):m.group(0) for m in re.finditer(self.pattern_class, code)}
        methods = [(m.start(0), m.end(0), m.group(0))  for m in re.finditer(self.pattern_method_all, code)]
        methods_names = re.findall(self.pattern_method_name, code)
        return classes, methods, methods_names


    def collect_method_info(self, code):
        classes, methods, methods_names = self.parse_code(code)
        sts = sorted(classes.keys())
        st = 0
        methods_info = {}
        for i, m in enumerate(methods):
            while m[0] > sts[st] and st + 1 < len(sts):
                st += 1
            full_name = classes[sts[st]] +': ' + methods_names[i]
            methods_info[full_name] =  (methods_names[i], m[2])
        return methods_info


    def find_difference(self, methods_info_a, methods_info_b):
        all_methods = list(methods_info_a.keys()) + list(methods_info_b.keys())
        changed_methods = set()

        for method in all_methods:
            if method in methods_info_a:
                method_name_a = methods_info_a[method][0]
                if method in methods_info_b:
                    method_code_a = methods_info_a[method][1]
                    method_code_b = methods_info_b[method][1]
                    if method_code_a != method_code_b:
                        changed_methods.add(method_name_a + " - changed")
                else:
                    changed_methods.add(method_name_a + " - added")
            else:
                method_name_b = methods_info_b[method][0]
                changed_methods.add(method_name_b + " - deleted")

        return changed_methods


    def open_repo(self, path='.'):
        try:
            self.repo = Repo(path, odbt=db.GitDB)
        except Exception:
            print("Check path to repository. Maybe, you should write path in double quotes\"\"")



    def find_changed_methods(self, path='.'):
        self.open_repo(path)
        code_a, code_b = self.collect_code_last_two_commits()
        code_b = self.remove_tabs(code_b)
        code_a = self.remove_tabs(code_a)
        methods_info_a = self.collect_method_info(code_a)
        methods_info_b = self.collect_method_info(code_b)
        self.changed_methods = self.find_difference(methods_info_a, methods_info_b)
        return self.find_difference(methods_info_a, methods_info_b)


if __name__ == "__main__":
    path = '.'
    if len(sys.argv) > 1:
            path = sys.argv[1]
            path = path.replace("\\", "/")
    cmf = ChangedMethodsFinder(path)
    print(cmf.find_changed_methods(path))


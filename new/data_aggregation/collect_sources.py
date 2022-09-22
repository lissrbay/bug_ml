import argparse
import os
import re
import base64

from git import Repo, db

from new.data_aggregation.changes.parser_java_kotlin import Parser
from new.constants import REPORTS_INTERMEDIATE_DIR
from new.data_aggregation.utils import iterate_reports
from new.data.report import Report, Frame, Code


def remove_tabs(code):
    code = re.sub(' +', ' ', code)
    return re.sub('\t+', '', code)


def code_fragment(bounds, code, offset=0):
    if not bounds:
        return ''
    if bounds[1] <= bounds[0]:
        return ''
    return " ".join(code.split('\n')[bounds[0] + offset: bounds[1]])


def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


def extract_method_code(code, method_name):
    parser = Parser()
    txt = remove_tabs(code)
    ast = parser.parse(txt)
    method_info = ast.get_method_names_and_bounds()
    code = ''
    bound = (0, 0)
    for method, scope in method_info.items():
        name = method.name
        bounds = scope.bounds
        name_ = name.split(':')[-1]
        if clean_method_name(method_name) in name_:
            line_bounds = find_file_lines(txt, bounds)
            method_code = code_fragment(line_bounds, txt, 0)
            code = method_code
            bound = line_bounds
    return code, bound


def get_file_by_commit(repo: Repo, commit: str, diff_file: str) -> str:
    code = repo.git.show('{}:{}'.format(commit, diff_file))
    return code


def find_file_lines(code: str, char_bounds):
    char_lines = []
    for i, line in enumerate(code.split('\n')):
        for _ in line:
            char_lines.append(i)
        else:
            char_lines.append(i)
    else:
        char_lines.append(i)

    assert len(code) == len(code)
    return char_lines[char_bounds[0]], char_lines[char_bounds[1]]


def get_method_from_code(code: str, method_name: str):
    method_code, bound = extract_method_code(code, method_name)
    hashed_code = base64.b64encode(method_code.encode('UTF-8'))
    return Code(begin=bound[0], end=bound[1], code=hashed_code)


def get_sources_for_report(repo: Repo, report: Report, commit: str) -> Report:
    frames_with_codes = []
    for frame in report.frames:
        if frame.path != '':
            diff_file = frame.path
            try:
                hashed_code = get_file_by_commit(repo, commit + "~1", diff_file)
                frame_code = get_method_from_code(hashed_code, frame.method_name)
                frames_with_codes.append(Frame(
                    code=frame_code,
                    method_name=frame.method_name,
                    file_name=frame.file_name,
                    line=frame.line,
                    path=frame.path,
                    label=frame.label,
                    file_path=frame.file_path,
                    has_recursion=frame.has_recursion
                ))
            except Exception:
                print(report.id, frame.file_name)
        else:
            frames_with_codes.append(frame)

    return Report(report.id, report.exceptions, report.hash, frames_with_codes)


def collect_sources_for_all_reports(repo: Repo, path_to_reports: str):
    reports_success = 0
    for file_name in iterate_reports(path_to_reports):
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_report(path_to_file)
        if report.hash == "":
            continue

        report = get_sources_for_report(repo, report, report.hash)
        report.save_report(path_to_file)
        reports_success += 1

    print(f"Successed collect code data for {reports_success} reports.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    # parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


def collect_sources_for_reports(repo_path: str, data_dir: str):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    collect_sources_for_all_reports(repo, path_to_reports)


if __name__ == "__main__":
    args = parse_args()
    collect_sources_for_reports(args.repo_path, args.data_dir)

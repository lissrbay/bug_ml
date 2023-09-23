import os
import subprocess
import argparse

import json

from new.data.report import Report
from new.data_aggregation.utils import iterate_reports
from new.data_aggregation.changes.get_all_changed_methods import get_all_changed_methods
from new.data_aggregation.match_reports_fixes import match_reports_to_labels
from new.data_aggregation.add_path_info import add_paths_to_reports
from new.data_aggregation.collect_sources import collect_sources_for_reports
from new.data_aggregation.git_data import add_git_data
#from new.data_aggregation.pycode2seq_embeddings import get_reports_embeddings
from new.constants import EMBEDDINGS_CACHE, REPORTS_INTERMEDIATE_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()

def dump_to_readable_format(data_dir, path_to_reports):
    for file_name in iterate_reports(path_to_reports):
        report_path = os.path.join(path_to_reports, file_name)
        report = Report.load_report(report_path)
        code_json, report_json = report.dump_to_code_and_stack()
        report_json_path = os.path.join(data_dir, 'last_reports', str(report.id) + '.report')
        code_json_path = os.path.join(data_dir, 'last_reports', str(report.id) + '.code')

        json.dump(report_json, open(report_json_path, "w"))
        json.dump(code_json, open(code_json_path, "w"))


if __name__ == "__main__":
    args = parse_args()
    collect_commits = ["sh", "../../collect_fix_commits.sh", args.repo_path, args.data_dir]
    subprocess.Popen(collect_commits).communicate()

    get_all_changed_methods(args.repo_path, args.reports_path, args.data_dir)
    match_reports_to_labels(args.reports_path,  args.data_dir)
    dump_to_readable_format(args.data_dir, os.path.join(args.data_dir, REPORTS_INTERMEDIATE_DIR))
    add_paths_to_reports(args.repo_path, args.data_dir, args.files_limit)
    dump_to_readable_format(args.data_dir, os.path.join(args.data_dir, REPORTS_INTERMEDIATE_DIR)) # here
    collect_sources_for_reports(args.repo_path, args.data_dir, args.files_limit)
    dump_to_readable_format(args.data_dir, os.path.join(args.data_dir, REPORTS_INTERMEDIATE_DIR))
    add_git_data(args.repo_path, args.data_dir, args.files_limit)
    get_reports_embeddings(args.reports_path, args.data_dir, EMBEDDINGS_CACHE, args.files_limit)


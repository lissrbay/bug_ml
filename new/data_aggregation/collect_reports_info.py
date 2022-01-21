import subprocess
import argparse

from new.data_aggregation.get_all_changed_methods import get_all_changed_methods
from new.data_aggregation.match_reports_fixes import match_reports_to_labels
from new.data_aggregation.add_path_info import add_paths_to_reports
from new.data_aggregation.collect_sources import collect_sources_for_reports
from new.data_aggregation.pycode2seq_embeddings import get_reports_embeddings

ISSUE_REPORTS_MAPPING_FILE = "issue_report_ids.csv"
INTELLIJ_CHANGED_METHODS_FILE = "fixed_methods.txt"
INTELLIJ_COMMIT_INFO = "commit_fix_hashes.txt"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--intellij_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_commits = ["sudo", "sh", "./collect_fix_commits.sh", args['intellij_path']]
    subprocess.Popen(collect_commits).communicate()

    get_all_changed_methods(args['intellij_path'], args['data_dir'])
    match_reports_to_labels(args['reports_path'], args['data_dir'])
    add_paths_to_reports(args['intellij_path'], args['reports_path'], args['files_limit'])
    collect_sources_for_reports(args['intellij_path'], args['reports_path'], args['files_limit'])
    get_reports_embeddings(args['reports_path'], args['data_dir'], args['files_limit'])

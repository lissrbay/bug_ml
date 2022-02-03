import subprocess
import argparse

from new.data_aggregation.get_all_changed_methods import get_all_changed_methods
from new.data_aggregation.match_reports_fixes import match_reports_to_labels
from new.data_aggregation.add_path_info import add_paths_to_reports
from new.data_aggregation.collect_sources import collect_sources_for_reports
from new.data_aggregation.git_data import add_git_data
from new.data_aggregation.pycode2seq_embeddings import get_reports_embeddings
from new.constants import CODE2SEQ_EMBS_CACHED

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_commits = ["sudo", "sh", "./collect_fix_commits.sh", args.repo_path]
    subprocess.Popen(collect_commits).communicate()

    get_all_changed_methods(args.repo_path, args.data_dir)
    match_reports_to_labels(args.reports_path,  args.data_dir)
    add_paths_to_reports(args.repo_path, args.reports_path, args.files_limit)
    collect_sources_for_reports(args.repo_path, args.reports_path, args.files_limit)
    add_git_data(args.repo_path, args.reports_path, args.files_limit)
    get_reports_embeddings(args.reports_path,  args.data_dir, CODE2SEQ_EMBS_CACHED, args.files_limit)

import subprocess
import argparse

from new.data_aggregation.get_all_changed_methods import get_all_changed_methods
from new.data_aggregation.match_reports_fixes import match_reports_to_labels
from new.data_aggregation.add_path_info import add_paths_to_reports
from new.data_aggregation.collect_sources import collect_reports_sources
from new.data_aggregation.git_data import add_git_data
from new.data_aggregation.pycode2seq_embeddings import get_reports_embeddings
from new.constants import EMBEDDINGS_CACHE


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_commits = ["sh", "../../collect_fix_commits.sh", args.repo_path, args.data_dir]
    subprocess.Popen(collect_commits).communicate()

    get_all_changed_methods(args.repo_path, args.data_dir)
    match_reports_to_labels(args.reports_path,  args.data_dir)
    add_paths_to_reports(args.repo_path, args.reports_path, args.files_limit)
    collect_reports_sources(args.repo_path, args.reports_path, args.files_limit)
    add_git_data(args.repo_path, args.reports_path, args.files_limit)
    get_reports_embeddings(args.reports_path, args.data_dir, EMBEDDINGS_CACHE, args.files_limit)

# --repo_path /Users/Aleksandr.Khvorov/jb/idea/intellij --reports_path /Users/Aleksandr.Khvorov/jb/exception-analyzer/data/intellij_fixed_201007 --data_dir /Users/Aleksandr.Khvorov/jb/exception-analyzer/bug_ml/data
# sudo sh collect_fix_commits.sh /Users/Aleksandr.Khvorov/jb/idea/intellij /Users/Aleksandr.Khvorov/jb/exception-analyzer/bug_ml/data

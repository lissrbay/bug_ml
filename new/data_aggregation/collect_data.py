import subprocess
import argparse

from new.data_aggregation.changes.save_changed_methods import save_changed_methods
from new.data_aggregation.match_reports_fixes import match_reports_to_labels
from new.data_aggregation.add_path_info import add_paths_to_reports
from new.data_aggregation.collect_sources import collect_reports_sources
from new.data_aggregation.git_data import add_git_data
from new.data_aggregation.pycode2seq_embeddings import get_reports_embeddings
from new.constants import EMBEDDINGS_CACHE
from new.data_aggregation.pipeline.save_fix_commits import save_fix_commits


def collect_data(repo_path: str, reports_dir: str, data_dir: str, files_limit: int):
    """
    Collects all necessary data.
    """

    save_fix_commits(repo_path, data_dir)  # Collecting fix commits to data directory
    save_changed_methods(repo_path, data_dir)

    match_reports_to_labels(reports_dir, data_dir)
    add_paths_to_reports(repo_path, reports_dir, files_limit)
    collect_reports_sources(repo_path, reports_dir, files_limit)
    add_git_data(repo_path, reports_dir, files_limit)
    get_reports_embeddings(reports_dir, data_dir, EMBEDDINGS_CACHE, files_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)
    args = parser.parse_args()

    collect_data(args.repo_path, args.reports_dir, args.data_dir, args.files_limit)

# --repo_path /Users/Aleksandr.Khvorov/jb/idea/intellij --reports_path /Users/Aleksandr.Khvorov/jb/exception-analyzer/data/intellij_fixed_201007 --data_dir /Users/Aleksandr.Khvorov/jb/exception-analyzer/bug_ml/data
# sudo sh collect_fix_commits.sh /Users/Aleksandr.Khvorov/jb/idea/intellij /Users/Aleksandr.Khvorov/jb/exception-analyzer/bug_ml/data

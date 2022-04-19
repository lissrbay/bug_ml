import argparse

from new.data_aggregation.pipeline.label_reports import label_reports
from new.data_aggregation.pipeline.save_changed_methods import save_changed_methods
from new.data_aggregation.pipeline.save_fix_commits import save_fix_commits
from new.data_aggregation.pipeline.save_issue_to_commits import save_issue_to_commits


def run(repo_path: str, reports_dir: str, issue_report_path: str, data_dir: str):
    save_fix_commits(repo_path, data_dir)
    save_issue_to_commits(data_dir)
    save_changed_methods(repo_path, data_dir)
    label_reports(reports_dir, issue_report_path, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--reports_dir", type=str)
    parser.add_argument("--issue_report_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    run(args.repo_path, args.report_dir, args.issue_report_path, args.data_dir)

import os
from argparse import ArgumentParser
from typing import Dict, Any

from git import Repo, db

from new.constants import FINAL_REPORTS_DIR, REPORTS_INTERMEDIATE_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports


def save_commit_file_info(repo: Repo, frame: Frame, file_path: str, commit_hash: str) -> Dict[str, Any]:
    revlist = [commit for commit in repo.iter_commits(rev=commit_hash, paths=file_path, max_count=1)]
    if len(revlist) == 0:
        return {}
    commit_data = revlist[0]
    frame_meta = frame.meta
    frame_meta['committed_date'] = commit_data.committed_date
    frame_meta['authored_date'] = commit_data.authored_date
    frame_meta['author'] = commit_data.author

    return frame_meta


def add_git_data_to_frames(repo: Repo, report: Report, frame_limit: int = 80) -> Report:
    commit_hash = report.hash
    frames = []
    for frame in report.frames[:frame_limit]:
        is_saved = False
        frame_meta = frame.meta
        if commit_hash and "path" in frame.meta and frame.meta['path'] != "":
            is_saved = save_commit_file_info(repo, frame, frame.meta['path'], commit_hash)
        if not is_saved:
            frame_meta['committed_date'] = 0
            frame_meta['authored_date'] = 0
            frame_meta['author'] = 'no_author'
        frames.append(Frame(frame.code, frame_meta))

    return Report(report.id, report.exceptions, report.hash, frames)


def add_git_data(repo_path: str, data_dir: str, frame_limit: int):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir, REPORTS_INTERMEDIATE_DIR)
    reports_save_path = os.path.join(data_dir, FINAL_REPORTS_DIR)
    os.makedirs(reports_save_path, exist_ok=True)
    for file_name in iterate_reports(path_to_reports):
        report_path = os.path.join(path_to_reports, file_name)
        report = Report.load_report(report_path)
        report = add_git_data_to_frames(repo, report, frame_limit)

        report_save_path = os.path.join(reports_save_path, file_name)
        report.save_report(report_save_path)

    print(f"Successfully collect git data for.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--frame_limit", type=int, default=80)

    args = parser.parse_args()
    add_git_data(args.repo_path, args.data_dir, args.files_limit)

import os
from argparse import ArgumentParser
from typing import Dict, Any

from git import Repo, db

from new.constants import FINAL_REPORTS_DIR
from new.data.report import Report, Frame
from new.data_aggregation.utils import iterate_reports
from multiprocessing import Pool

def save_commit_file_info(repo: Repo, file_path: str, commit_hash: str, method_begin: int, method_end: int) -> Dict[str, Any]:
    ts = []
    authors = []
    code = []
    try:
        for commit, lines in repo.blame(rev=commit_hash, file=file_path):
            ts.extend([commit.authored_date] * len(lines))
            authors.extend([commit.author.email] * len(lines))
            code.extend(lines)
    except Exception:
        return (ts, authors)

    return (ts[method_begin: method_end+1], authors[method_begin: method_end+1])


def add_git_data_to_frames(repo: Repo, report: Report) -> Report:
    commit_hash = report.hash
    frames = []
    pool = Pool()
    local_cash = {}
    parallel_processing = []
    parallel_processing_paths = []
    for frame in report.frames:
        if commit_hash and "path" in frame.meta and frame.meta['path'] != "" and frame.code.end > 0:
            if not (frame.meta['path'] in local_cash):
                parallel_processing.append((repo, frame.meta['path'], commit_hash, frame.code.begin, frame.code.end))
                local_cash[frame.meta['path']] = 1
                parallel_processing_paths.append(frame.meta['path'])
            #is_saved = save_commit_file_info(repo, frame, frame.meta['path'], commit_hash)

    ans = pool.starmap(save_commit_file_info, parallel_processing)
    ans = {k: v for k, v in zip(parallel_processing_paths, ans)}

    for frame in report.frames:
        frame_meta = frame.meta
        if commit_hash and "path" in frame.meta and frame.meta['path'] != "" and frame.code.end > 0:
            ts, authors = ans[frame.meta['path']]
            frame_meta['ts'] = ts
            frame_meta['authors'] = authors
        else:
            frame_meta['ts'] = []
            frame_meta['authors'] = []
        frames.append(Frame(frame.code, frame_meta))

    return Report(report.id, report.exceptions, report.hash, frames)


def add_git_data(repo_path: str, data_dir: str, frame_limit: int):
    repo = Repo(repo_path, odbt=db.GitDB)

    path_to_reports = os.path.join(data_dir)#, REPORTS_INTERMEDIATE_DIR)
    reports_save_path = os.path.join(data_dir, FINAL_REPORTS_DIR)
    os.makedirs(reports_save_path, exist_ok=True)

    for file_name in iterate_reports(path_to_reports):
        report_path = os.path.join(path_to_reports, file_name)
        report = Report.load_report(report_path)
        report = add_git_data_to_frames(repo, report)

        report_save_path = os.path.join(reports_save_path, file_name)
        report.save_report(report_save_path)

    print(f"Successfully collect git data for.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--frame_limit", type=int, default=80)

    args = parser.parse_args()
    add_git_data(args.repo_path, args.data_dir, args.frame_limit)

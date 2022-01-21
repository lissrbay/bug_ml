from get_java_methods import ChangedMethodsFinder
import json
import pandas as pd
from git import Repo, db
import tqdm
import os
from new.data.report import Report, Frame
from argparse import ArgumentParser


class GitFeaturesExtractor:
    def __init__(self, path_to_repo: str, path_to_reports: str, frame_limit: int = 80):
        self.path_to_repo = path_to_repo
        self.path_to_reports = path_to_reports
        self.repo = self.open_repo(path_to_repo)
        self.frame_limit = frame_limit

    def open_repo(self, path: str):
        try:
            repo = Repo(path, odbt=db.GitDB)
        except Exception:
            print("Check path to repository. Maybe, you should write path in double quotes\"\"")
        return repo

    def save_commit_file_info(self, frame: Frame, file_path: str, commit_hash: str) -> bool:
        revlist = [commit for commit in self.repo.iter_commits(rev=commit_hash, paths=file_path, max_count=1)]
        if len(revlist) == 0:
            return {}
        commit_data = revlist[0]
        frame_meta = frame.meta
        frame_meta['committed_date'] = commit_data.committed_date
        frame_meta['authored_date'] = commit_data.authored_date
        frame_meta['author'] = commit_data.author

        return frame_meta

    def add_git_data_to_frames(self, report: Report) -> Report:
        commit_hash = report.hash
        frames = []
        for frame in report.frames[:self.frame_limit]:
            is_saved = False
            if commit_hash and "path" in frame.meta and frame.meta['path'] != "":
                is_saved = self.save_commit_file_info(frame, frame.meta['path'], commit_hash)
            if not is_saved:
                frame_meta = frame.meta
                frame_meta['committed_date'] = 0
                frame_meta['authored_date'] = 0
                frame_meta['author'] = 'no_author'
            frames.append(Frame(frame.code, frame_meta))

        return Report(report.id, report.exceptions, report.hash, frames)

    def iterate_reports(self):
        for root, _, files in filter(lambda x: (x[0] == self.path_to_reports), os.walk(self.path_to_reports)):
            for file in tqdm.tqdm(files):
                report_path = os.path.join(self.path_to_reports, file)
                report = Report.load_report(report_path)
                report = self.add_git_data_to_frames(report)

                report.save_report(report_path)


def add_git_data(intellij_path: str, reports_path: str, frame_limit: int):
    gfe = GitFeaturesExtractor(intellij_path, reports_path, frame_limit)
    gfe.iterate_reports()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--intellij_path", type=str)
    parser.add_argument("--frame_limit", type=int, default=80)

    args = parser.parse_args()
    main(args.intellij_path, args.reports_path, args.files_limit)

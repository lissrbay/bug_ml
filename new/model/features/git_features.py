from typing import List

from torch import Tensor, FloatTensor

from new.data.report import Report
import pandas as pd

from new.model.report_encoders.report_encoder import ReportEncoder


class GitFeaturesTransformer(ReportEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.data = None

    def collect_to_df(self, reports: List[Report], target: List[List[int]]):
        label, commit_time, modified_time_diff, authors, report_frame = [], [], [], [], []
        for j, report in enumerate(reports):
            for i, frame in enumerate(report.frames):
                commit_time.append(frame.meta['committed_date'] if 'commited_date' in frame.meta else 0)
                modified_time_diff.append(frame.meta['committed_date'] - frame.meta['authored_date']
                                            if 'commited_date' in frame.meta and 'authored_date' in frame.meta else 0)
                authors.append(frame.meta['author'].email if 'author' in frame.meta else '')

                report_frame.append(i)
                label.append(target[j][i])
        return pd.DataFrame({'commit_time': commit_time,
            'modified_time_diff': modified_time_diff,
            'author':authors,
            'method_stack_position':report_frame,
            'label':label})

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CodeFeatures':
        self.data = self.collect_to_df(reports, target)

        self.author_occurences = self.compute_author_occurences()
        self.authors_bug_stats = self.compute_author_bugs()
        self.author_lifetime = self.compute_author_lifetime()
        return self


    def compute_author_occurences(self):
        author_occurrences = self.data.groupby('author').count()['label'].reset_index()
        author_occurrences.columns = ['author', 'occurences']
        return author_occurrences

    def compute_author_bugs(self):
        bugs_in_reports = self.data.groupby('author').sum()['label'].reset_index()
        bugs_in_reports.columns = ['author', 'bugs_count']
        author_occurrences = self.author_occurences

        authors_bug_stats = bugs_in_reports.merge(author_occurrences, on='author', how='inner')
        laplace_const = 0.3
        authors_bug_stats['bugs_to_occurences'] = (authors_bug_stats['bugs_count'] + laplace_const) / (
                    authors_bug_stats['occurences'] + 1)

        return authors_bug_stats

    def compute_author_lifetime(self):
        author_lifetime = self.data.groupby('author').min()['commit_time'].reset_index()
        author_lifetime.columns = ['author', 'first_commit_time']

        author_last_commit = self.data.groupby('author').max()['commit_time'].reset_index()
        author_last_commit.columns = ['author', 'last_commit_time']

        author_lifetime = author_lifetime.merge(author_last_commit, on='author', how='inner')
        author_lifetime['worktime'] = author_last_commit['last_commit_time'] - author_lifetime['first_commit_time']
        return author_lifetime

    def get_author_features(self, frame):
        author_lifetime = self.author_lifetime
        method_author = frame.meta['author'].email if 'author' in frame.meta else ''
        worktime = author_lifetime[author_lifetime.author == method_author]['worktime'].values[0]
        last_commit_time = author_lifetime[author_lifetime.author == method_author]['last_commit_time'].values[0]

        author_occurences = self.author_occurences
        occurences = author_occurences[author_occurences.author == method_author]['occurences'].values[0]

        authors_bug_stats = self.authors_bug_stats
        bug_count = authors_bug_stats[authors_bug_stats.author == method_author]['bugs_count'].values[0]
        bugs_to_occurences = authors_bug_stats[authors_bug_stats.author == method_author]['bugs_to_occurences'].values[0]

        return [worktime, last_commit_time, occurences, bug_count, bugs_to_occurences]

    def encode_report(self,  report: Report) -> Tensor:
        report_features = []
        for i, frame in enumerate(report.frames[:self.frames_count]):
            modified_time_diff = (frame.meta['committed_date'] - frame.meta['authored_date']
                                        if 'commited_date' in frame.meta and 'authored_date' in frame.meta else 0)
            author_features = self.get_author_features(frame)
            report_features.append(author_features + [modified_time_diff])

        for i in range(self.frames_count - len(report.frames)):
            report_features.append([0 for j in range(6)])

        return FloatTensor(report_features)

    @property
    def dim(self) -> int:
        return 6

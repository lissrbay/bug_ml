from typing import List

from new.data.report import Report
from new.model.features.feature import BaseFeature
from data_aggregation.add_code_data import get_report_code
import base64
import re
import pandas as pd

class GitFeaturesTransformer(BaseFeature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.commit_time = []
        self.modified_time_diff = []
        self.authors = []

        self.report_id = []
        self.report_frame = []

    def collect_to_df(self, reports: List[Report]):
        for report in reports:
            for i, frame in enumerate(report.frames):
                self.commit_time.append(frame.meta['commit_time'])
                self.modified_time_diff.append(frame.meta['modified_time_diff'])
                self.authors.append(frame.meta['author'])

                self.report_id.append(report.id)
                self.report_frame.append(i)

        return pd.DataFrame({'commit_time': self.commit_time,
            'modified_time_diff': self.modified_time_diff,
            'author':self.authors,
            'report_id':self.report_id,
            'method_stack_position':self.report_frame})

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CodeFeatures':
        data = self.collect_to_df(reports)

        self.author_occurences = self.compute_author_occurences_count(data)
        self.authors_bug_stats = self.compute_author_bugs(data)
        self.author_lifetime = self.compute_author_lifetime(data)


    def compute_author_occurences(self, data: pd.DataFrame):
        author_occurrences = data.groupby('author').count()['label'].reset_index()
        author_occurrences.columns = ['author', 'occurences']
        return author_occurrences

    def compute_author_bugs(self, data: pd.DataFrame):
        bugs_in_reports = data.groupby('author').sum()['label'].reset_index()
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
        method_author = frame.meta['author']
        worktime = author_lifetime[author_lifetime.author == method_author]['worktime'].values[0]
        last_commit_time = author_lifetime[author_lifetime.author == method_author]['last_commit_time'].values[0]

        author_occurences = self.author_occurences
        occurences = author_occurences[author_occurences.author == method_author]['occurences'].values[0]

        authors_bug_stats = self.author_bugs
        bug_count = authors_bug_stats[authors_bug_stats.author == method_author]['bugs_count'].values[0]
        bugs_to_occurences = authors_bug_stats[authors_bug_stats.author == method_author]['bugs_to_occurences'].values[0]

        return [worktime, last_commit_time, occurences, bug_count, bugs_to_occurences]

    def transform(self, report: Report) -> List[List[float]]:
        report_features = []
        pos_by_times = []
        for i, frame in enumerate(report.frames):
            modified_time_diff = frame.meta['modified_time_diff']
            author_features = self.get_author_features(frame)
            pos_by_time = pos_by_times[i]
            report_features.append(author_features + [modified_time_diff] + pos_by_time)

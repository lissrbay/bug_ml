from typing import List

import torch
from torch import Tensor

from new.data.report import Report
import pandas as pd

from new.model.report_encoders.report_encoder import ReportEncoder


class GitFeaturesTransformer(ReportEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.frames_count = kwargs['frames_count']
        self.author_occurences: pd.DataFrame = None
        self.authors_bug_stats: pd.DataFrame = None
        self.author_lifetime: pd.DataFrame = None

        self.raw_feature_names = [
            'commit_time',
            'modified_time_diff',
            'authors',
            'report_frame',
            'label'
        ]
        self.feature_names = [
                         'bugs_to_occurences',
                         'worktime',
                         'last_commit_time',
                         'occurences',
                         'bug_count',
                         'modified_time_diff'
                         ]

    def get_modified_time(self, frame):
        committed_date = frame.meta['committed_date'] if 'commited_date' in frame.meta else 0
        authored_date = frame.meta['authored_date'] if 'authored_date' in frame.meta else 0
        return committed_date, authored_date

    def collect_to_df(self, reports: List[Report], target: List[List[int]]):
        features = {k: [] for k in self.raw_feature_names}
        for j, report in enumerate(reports):
            for i, frame in enumerate(report.frames):
                committed_date, authored_date = self.get_modified_time(frame)
                authors = frame.meta['author'].email if 'author' in frame.meta else ''

                features['commit_date'].append(committed_date)
                features['modified_time_diff'].append(committed_date - authored_date)
                features['authors'].append(authors)
                features['label'].append(target[j][i])

        return pd.DataFrame(features)

    def fit(self, reports: List[Report], target: List[List[int]]) -> 'CodeFeatures':
        data = self.collect_to_df(reports, target)
        self.author_occurences = self.compute_author_occurences(data)
        self.authors_bug_stats = self.compute_author_bugs(data)
        self.author_lifetime = self.compute_author_lifetime(data)
        return self


    def compute_author_occurences(self, data):
        author_occurrences = data.groupby('author').count()['label'].reset_index()
        author_occurrences.columns = ['author', 'occurences']
        return author_occurrences

    def compute_author_bugs(self, data):
        bugs_in_reports = data.groupby('author').sum()['label'].reset_index()
        bugs_in_reports.columns = ['author', 'bugs_count']
        author_occurrences = self.author_occurences

        authors_bug_stats = bugs_in_reports.merge(author_occurrences, on='author', how='inner')
        laplace_const = 0.3
        authors_bug_stats['bugs_to_occurences'] = (authors_bug_stats['bugs_count'] + laplace_const) / (
                    authors_bug_stats['occurences'] + 1)

        return authors_bug_stats

    def compute_author_lifetime(self, data):
        author_lifetime = data.groupby('author').min()['commit_time'].reset_index()
        author_lifetime.columns = ['author', 'first_commit_time']

        author_last_commit = data.groupby('author').max()['commit_time'].reset_index()
        author_last_commit.columns = ['author', 'last_commit_time']

        author_lifetime = author_lifetime.merge(author_last_commit, on='author', how='inner')
        author_lifetime['worktime'] = author_last_commit['last_commit_time'] - author_lifetime['first_commit_time']
        return author_lifetime

    def extract_frames_authors(self, frames):
        authors = []
        for frame in frames:
            method_author = frame.meta['author'].email if 'author' in frame.meta else ''
            authors.append(method_author)

        return pd.DataFrame({'author': authors})

    def get_author_features(self, frames, features):
        frames_authors = self.extract_frames_authors(frames)

        author_lifetime = self.author_lifetime
        author_lifetime = author_lifetime.merge(frames_authors, on='author', how='right')

        author_occurences = self.author_occurences
        occurences = author_occurences.merge(frames_authors, on='author', how='right')

        authors_bug_stats = self.authors_bug_stats
        authors_bug_stats = authors_bug_stats.merge(frames_authors, on='author', how='right')

        features['bugs_count'] = authors_bug_stats['bugs_count']
        features['bugs_to_occurences'] = authors_bug_stats['bugs_to_occurences']
        features['occurences'] = occurences['occurences']
        features['worktime'] = author_lifetime['worktime']
        features['last_commit_time'] = author_lifetime['last_commit_time']

        return features

    def get_modified_times(self, frames, features):
        for frame in frames:
            committed_date, authored_date = self.get_modified_time(frame)
            features['modified_time_diff'].append(committed_date - authored_date)
        return features

    def encode_report(self,  report: Report) -> Tensor:
        features = {k: [] for k in self.feature_names}

        features = self.get_modified_times(report.frames[:self.frames_count], features)
        features = self.get_author_features(report.frames[:self.frames_count], features)

        report_features = [torch.FloatTensor(features[name]) for name in self.feature_names]
        report_features = torch.cat(report_features, dim=0).T

        return report_features

    @property
    def dim(self) -> int:
        return len(self.features_names)

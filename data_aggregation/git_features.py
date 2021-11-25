from get_java_methods import ChangedMethodsFinder 
import json
import pandas as pd
from git import Repo, db
import tqdm
import os

class GitFeaturesExtractor:
    def __init__(self, path_to_repo, path_to_reports):
        self.commit_time = []
        self.modified_time_diff = []
        self.authors = []
        self.report_id = []
        self.report_frame = []
        self.path_to_repo = path_to_repo
        self.path_to_reports = path_to_reports
        self.repo = self.open_repo(path_to_repo)

    def open_repo(self, path='.'):
        try:
            repo = Repo(path, odbt=db.GitDB)
        except Exception:
            print("Check path to repository. Maybe, you should write path in double quotes\"\"")
        return repo

    def read_report(self, path_to_report):
        report_path = open(path_to_report, 'r')
        try:
            report = json.load(report_path)
            return report
        except Exception:
            print("oh shit there's no data")
        report_path.close()
        return {}

    def save_commit_file_info(self, file_path, commit_hash):
        revlist = [commit for commit in self.repo.iter_commits(rev=commit_hash, paths=file_path, max_count=1)]
        if (revlist[0] == commit_hash):
            print("me")
        if len(revlist) == 0:
            return False
        commit_data = revlist[0]
        self.commit_time.append(commit_data.committed_date)
        self.modified_time_diff.append(commit_data.authored_date)
        self.authors.append(commit_data.author)
        return True
        

    def iterate_reports(self):
        for root, _, files in os.walk(path_to_reports):
            if not (root == path_to_reports):
                continue
            for file in tqdm.tqdm(files):
                report_path = os.path.join(path_to_reports, file)
                report = self.read_report(report_path)
                commit_hash = report['hash'] if "hash" in report else ""
                if not ("frames" in report):
                    continue
                
                for i, frame in enumerate(report['frames'][:80]):
                    is_saved = False
                    if commit_hash and "path" in frame and frame['path'] != "":
                        is_saved = self.save_commit_file_info(frame['path'], commit_hash)
                    if not is_saved:
                        self.commit_time.append(0)
                        self.modified_time_diff.append(0)
                        self.authors.append("no_author")

                    self.report_id.append(file.split('.')[0])
                    self.report_frame.append(i)


    def save_features(self, save_path):
        df = pd.DataFrame({'commit_time': self.commit_time,
        'modified_time_diff': self.modified_time_diff,
        'author':self.authors,
        'report_id':self.report_id,
        'method_stack_position':self.report_frame})

        df.to_csv(save_path)

    
class AuthorFeaturesExtractor:
    def __init__(self, git_features_path, features_path):
        self.git_features_path = git_features_path
        self.features_path = features_path
        self.git_features = pd.read_csv(self.git_features_path, index_col=0)
        self.features = pd.read_csv(self.features_path, index_col=0)
        self.data  = self.features.merge(self.git_features, on=['report_id', 'method_stack_position'])

    def author_occurencies(self):
        return self.data.groupby('author').count()['label'].reset_index()

    def author_bugs(self):
        bugs_in_reports = self.data.groupby('author').sum()['label'].reset_index()
        bugs_in_reports.columns = ['author', 'bugs_count']
        author_occurencies = self.author_occurencies()
        author_occurencies.columns = ['author', 'occurencies']

        authors_bug_stats = bugs_in_reports.merge(author_occurencies, on='author', how='inner')
        laplace_const = 0.3
        authors_bug_stats['bugs_to_occurencies'] = (authors_bug_stats['bugs_count'] + laplace_const) / (authors_bug_stats['occurencies'] + laplace_const)

        return authors_bug_stats

    def author_lifetime(self):
        author_lifetime = self.data.groupby('author').min()['commit_time'].reset_index()
        author_lifetime.columns = ['author', 'first_commit_time']

        author_last_commit = self.data.groupby('author').max()['commit_time'].reset_index()
        author_last_commit.columns = ['author', 'last_commit_time']

        author_lifetime = author_lifetime.merge(author_last_commit, on='author', how='inner')
        author_lifetime['worktime'] = author_last_commit['last_commit_time'] - author_lifetime['first_commit_time']
        return author_lifetime

    def collect_author_features(self):
        author_lifetime = self.author_lifetime()
        authors_bug_stats = self.author_bugs()
        report_author_files = self.report_author_files()
        return author_lifetime.merge(authors_bug_stats, on='author', how='inner').merge(report_author_files, on=['author'], how='inner')

    def report_author_files(self):
        report_author_files = self.data.groupby(['author', 'report_id']).count()['label'].reset_index()
        report_author_files.columns = ['author', 'report_id', 'author_report_changed_files']
        return report_author_files

    def save_features(self, save_path):
        df = afe.collect_author_features()
        df.to_csv(save_path)

if __name__ == "__main__":
    path_to_reports = ''
    intellij_path = ''
    gfe = GitFeaturesExtractor(intellij_path, path_to_reports)
    gfe.iterate_reports()
    gfe.save_features('../data/git_features.csv')
    afe = AuthorFeaturesExtractor('../data/git_features.csv', '../data/reports_features_20211114.csv')
    afe.save_features('../data/authors_features.csv')
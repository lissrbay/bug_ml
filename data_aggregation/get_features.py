import re
import json
import os
from typing import Dict, Any

from tqdm import tqdm
import pickle
import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.is_java_standart = []
        self.method_len = []
        self.method_tokens_count = []
        self.method_stack_position = []
        self.method_file_position = []
        self.exception_class = []
        self.has_runs = []
        self.has_dollars = []
        self.method_name = []
        self.label = []
        self.has_no_code = []
        self.newObjectsCount = []
        self.is_parallel = []
        self.report_id = []

    @staticmethod
    def load_code_data(path_to_code: str, path_to_report_ids: str):
        reports_code = pickle.load(open(path_to_code, "rb"))
        reports_ids = pickle.load(open(path_to_report_ids, "rb"))
        return reports_code, reports_ids

    def get_features_from_files(self, path_to_reports, path_to_code, path_to_report_ids):
        reports_code, reports_ids = self.load_code_data(path_to_code, path_to_report_ids)
        for report_id in tqdm(reports_ids):
            report_path = open(os.path.join(path_to_reports, str(report_id) + ".json"), 'r')
            report = json.load(report_path)
            for i, frame in enumerate(report['frames'][:80]):
                frame['class'] = report['class']
                frame['pos'] = i
                frame['id'] = report_id
                self.add_feature_from_code(reports_code[report_id][i])
                self.add_feature_from_metadata(frame)

    def add_feature_from_code(self, method_code: str) -> 'FeatureExtractor':
        self.method_tokens_count.append(len(method_code.split()))
        self.newObjectsCount.append(len(re.findall(r' new ', method_code)))
        self.has_no_code.append(len(method_code) == 0)
        self.method_len.append(len(method_code))
        return self

    def add_feature_from_metadata(self, method_meta: Dict[str, Any]) -> 'FeatureExtractor':
        self.exception_class.append(method_meta['class'])
        method_name = method_meta['method_name']
        self.has_runs.append("run" in method_name)
        self.has_dollars.append("$" in method_name)
        self.is_parallel.append("Thread" in method_name)
        self.method_file_position.append(int(method_meta['line_number']))
        self.is_java_standart.append(method_name[:4] == 'java')
        self.method_stack_position.append(method_meta['pos'])
        self.label.append(method_meta['label'])
        self.report_id.append(method_meta['id'] if 'id' in method_meta else 0)
        self.method_name.append(method_name)
        return self

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame({"is_java_standart": self.is_java_standart,
                           'method_len': self.method_len,
                           'method_tokens_count': self.method_tokens_count,
                           'method_stack_position': self.method_stack_position,
                           'method_file_position': self.method_file_position,
                           'exception_class': self.exception_class,
                           'has_runs': self.has_runs,
                           'has_dollars': self.has_dollars,
                           'method_name': self.method_name,
                           'label': self.label,
                           'has_no_code': self.has_no_code,
                           'newObjectsCount': self.newObjectsCount,
                           'is_parallel': self.is_parallel,
                           "report_id": self.report_id})
        return df

    def save_features(self, save_path):
        if not self.exception_class:
            print("You should run get features methods first.")
            return
        self.to_pandas().to_csv(save_path)

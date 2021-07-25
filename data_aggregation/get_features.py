import re
import json
import os
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

    def load_code_data(self, path_to_code, path_to_report_ids):
        self.reports_code=pickle.load(open(path_to_code, "rb"))
        self.reports_ids=pickle.load(open(path_to_report_ids, "rb"))
        

    def get_features_from_files(self, path_to_reports, path_to_code, path_to_report_ids):
        self.load_code_data(path_to_code, path_to_report_ids)
        for report_id in tqdm(self.reports_ids):
            report_path = open(os.path.join(path_to_reports, str(report_id)+".json"), 'r')
            report = json.load(report_path)
            self.report_id.append(report_id)
            for i, frame in enumerate(report['frames'][:80]):
                self.method_len.append(len(self.reports_code[report_id][i]))
   
                frame['class'] = report['class']
                frame['pos'] = i
                self.get_feature_from_code(self.reports_code[report_id][i])
                self.get_feature_from_metadata(frame)
        

    def get_feature_from_code(self, method_code):
        self.method_tokens_count.append(len(method_code.split()))
        self.newObjectsCount.append(len(re.findall(r' new ', method_code)))
        self.has_no_code.append(len(method_code) == 0)

    
    def get_feature_from_metadata(self, method_meta):
        self.exception_class.append(method_meta['class'])
        method_name = method_meta['method_name']
        self.has_runs.append("run" in method_name)
        self.has_dollars.append("$" in method_name)
        self.is_parallel.append("Thread"  in method_name)
        self.is_java_standart.append(method_name[:4] == 'java')
        self.method_stack_position.append(method_meta['pos'])
        self.label.append(method_meta['label'])
        if 'id' in method_meta:
            self.report_id.append(method_meta['id'])


    def to_pandas(self):
        df = pd.DataFrame({ "is_java_standart":self.is_java_standart,
            'method_len':self.method_len,
            'method_tokens_count':self.method_tokens_count,
            'method_stack_position':self.method_stack_position,
            'method_file_position':self.method_file_position,
            'exception_class':self.exception_class,
            'has_runs':self.has_runs,
            'has_dollars':self.has_dollars,
            'method_name':self.method_name,
            'label':self.label,
            'has_no_code':self.has_no_code,
            'newObjectsCount':self.newObjectsCount,
            'is_parallel':self.is_parallel})
        return df


    def save_features(self, save_path):
        if self.exception_class == []:
            print("You should run get features methods first.")
            return
        self.df = self.to_pandas()
        self.df.to_csv(save_path)

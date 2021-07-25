from common import Common
from extractor import Extractor
import os
import pandas as pd
from tqdm import tqdm
SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
EXTRACTION_API = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model, reports_path):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)
        self.reports_path = reports_path

    @staticmethod
    def read_file(input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()


    def model_prediction_df(self, input_filename):
        predict_lines, _, method_names = self.path_extractor.extract_paths(input_filename)
        if not predict_lines:
            return None
        _, embeddings = self.model.predict(predict_lines)
        method_vectors = []
        for i in range(len(method_names)):
            embedding = embeddings[i][0]
            method_vectors.append(embedding)
        df = pd.DataFrame(data=method_vectors)
        df['method'] = method_names
        return df


    def predict(self):
        print('Starting file processing...')
        path_to_report = self.reports_path
        c = 0
        for root, _, files in tqdm(os.walk(path_to_report)):
            if not (root.split('/')[-1]).isnumeric():
                continue
            for file in sorted(files):
                if not file.endswith('.java'):
                    continue
                input_filename = os.path.join(root,  file)
                df = self.model_prediction_df(input_filename)
                if df is None:
                    continue
                c += 1
                df.to_csv(input_filename[:len(input_filename)-5] + '.csv')
                print( str(c) + " " + input_filename + " - done")
            else:
                c += 1


    def create_temp_file(self, code):
        file_with_code = open("tmp.java", "w")
        file_with_code.write(code)
        file_with_code.close()


    def predict_by_code(self, code):
        self.create_temp_file(code)
        df = self.model_prediction_df("tmp.java")
        return df


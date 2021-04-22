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

    def predict(self):
        print('Starting file processing...')
        path_to_report = os.path.join(reports_path, "labeled_reports")
        c = 0
        for root, _, files in tqdm(os.walk(path_to_report)):
            if not (root.split('/')[-1]).isnumeric():
                continue
            for file in sorted(files):
                if not file.endswith('.java'):
                    continue
                input_filename = os.path.join(root,  file)
                predict_lines, _, method_names = self.path_extractor.extract_paths(input_filename)
                if not predict_lines:
                    print("oops")
                    continue
                _, embeddings = self.model.predict(predict_lines)
                method_vectors = []
                for i in range(len(method_names)):
                    embedding = embeddings[i][0]
                    method_vectors.append(embedding)
                df = pd.DataFrame(data=method_vectors)
                df['method'] = method_names
                df.to_csv(input_filename.split('.')[0] + '.csv')
                print(str(c) + " " + input_filename + " - done")
            else:
                c += 1

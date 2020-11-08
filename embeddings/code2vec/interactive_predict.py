import traceback

from common import common
from extractor import Extractor
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self):
        print('Starting file processing...')
        path_to_report = "labeled_reports"
        c = 0
        for root, dirs, files in os.walk(path_to_report):
            for file in files:
                if not file.endswith('.java'):
                    break
                input_filename = root + "/" + file
                try:
                    predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
                except ValueError as e:
                    print(e)
                    continue
                raw_prediction_results = self.model.predict(predict_lines)
                method_prediction_results = common.parse_prediction_results(
                    raw_prediction_results, hash_to_string_dict,
                    self.model.vocabs.target_vocab.special_words)
                methods_names = []
                methods_vectors = []
                for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
                    methods_names.append(''.join(method_prediction.original_name.split('|')))

                    if self.config.EXPORT_CODE_VECTORS:
                        methods_vectors.append(raw_prediction.code_vector)

                df = pd.DataFrame(data=methods_vectors)
                df['method'] = methods_names
                df.to_csv(input_filename.split('.')[0] + '.csv')
                print(input_filename + " - done")


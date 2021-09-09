import logging
import json
from pycode2seq import Code2Seq
import numpy as np
from tqdm import tqdm
import os
import argparse

model = Code2Seq.load("java")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
EMBEDDING_SIZE = 320

def get_embedding(self, filename, method_name):
    embeddings = []

    method_embeddings = model.methods_embeddings(filename)
    method_short_name = method_name.split('.')[-1]
    if method_short_name in method_embeddings:
        return method_embeddings[method_short_name].detach().numpy()
    else:
        return np.zeros(EMBEDDING_SIZE)

def get_reports_embeddings(path_to_report, files_limit=80):        
    preprocessed_count = 0
    embeddings = []
    for root, _, files in tqdm(os.walk(path_to_report)):
        if not (root.split('/')[-1]).isnumeric():
            continue
        report_path = root + ".json"
        report = json.load(open(report_path, "r"))
        for frame in report['frames'][:files_limit]:
            if os.path.exist(frame['file_name']):
                input_filename = os.path.join(root,  frame['file_name'])
                embedding = get_embedding(input_filename, method_name)
            else:
                embedding = np.zeros(EMBEDDING_SIZE)
            embeddings.append(embedding)
            preprocessed_count += 1
            logging.info(str(preprocessed_count) + " " + input_filename + " - done")
        for i in range(len(report['frames']), frames_limit):
            embeddings.append(embedding)
        report.close()
    np.save(os.path.join(save_dir, "X(pycode2seq).npy"), embeddings.reshape(-1, 80, EMBEDDING_SIZE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default='../data/')
    args = parser.parse_args()
    get_reports_embeddings(args.reports_path, args.save_dir)
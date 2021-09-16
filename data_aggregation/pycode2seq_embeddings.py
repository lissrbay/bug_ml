import logging
import json
from pycode2seq import Code2Seq
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
model = Code2Seq.load("java")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
EMBEDDING_SIZE = 320

def get_file_embeddings(filename):
    method_embeddings = model.methods_embeddings(filename)
    return method_embeddings

def get_method_embedding(method_embeddings, method_name):
    method_short_name = method_name.split('.')[-1]
    if method_short_name in method_embeddings:
        return method_embeddings[method_short_name].detach().numpy()
    else:
        return np.zeros(EMBEDDING_SIZE)


def get_reports_embeddings(path_to_report, save_dir, files_limit=80):        
    preprocessed_count = 0
    embeddings = []
    for root, _, files in tqdm(os.walk(path_to_report)):
        if not (root.split('/')[-1]).isnumeric():
            continue
        report_path = root + ".json"
        report = json.load(open(report_path, "r"))
        cashed_preds = {}
        labels = []
        for frame in report['frames'][:files_limit]:
            input_filename = os.path.join(root,  "" if frame['file_name'] is None else frame['file_name'])

            if os.path.isfile(input_filename):
                if input_filename in cashed_preds:
                    method_embeddings = cashed_preds[input_filename]
                else:
                    method_embeddings = get_file_embeddings(input_filename)
                    cashed_preds[input_filename] = method_embeddings
                embedding = get_method_embedding(method_embeddings, frame['method_name'])
            else:
                embedding = np.zeros(EMBEDDING_SIZE)
            embeddings.append(embedding)
            preprocessed_count += 1
            labels.append(frame['label'])
            logging.info(str(preprocessed_count) + " " + input_filename + " - done")
        for i in range(len(report['frames']), 80):
            embeddings.append(np.zeros(EMBEDDING_SIZE))
            labels.append(0)
        break
    np.save(os.path.join(save_dir, "X(pycode2seq).npy"), np.array(embeddings).reshape(-1, 80, EMBEDDING_SIZE))
    np.save(os.path.join(save_dir, "y(pycode2seq).npy"), np.array(labels).reshape(-1, 80))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default='../data/')
    args = parser.parse_args()
    get_reports_embeddings(args.reports_path, args.save_dir)
import argparse
import os
import pickle
from typing import Dict

import torch
from pycode2seq import Code2Seq

from new.constants import EMBEDDING_DIM, EMBEDDING_TMP_FILE, FINAL_REPORTS_DIR
from new.data.report import Report
from new.data_aggregation.utils import iterate_reports


def zeros() -> torch.Tensor:
    return torch.zeros(EMBEDDING_DIM)


def get_method_embedding(file_embeddings: Dict[str, torch.Tensor], method_name: str) -> torch.Tensor:
    method_short_name = method_name.split('.')[-1]
    if method_short_name in file_embeddings:
        return file_embeddings[method_short_name]
    else:
        return zeros()


def get_file_embeddings(model: Code2Seq, code: str) -> Dict[str, torch.Tensor]:
    if code:
        with open(EMBEDDING_TMP_FILE, 'w') as tmp_file:
            tmp_file.write(code)
        return model.methods_embeddings(EMBEDDING_TMP_FILE)
    else:
        return {}


def get_all_embeddings(data_dir: str, embs_name: str, files_limit: int = 80):
    embeddings = {}
    file_embeddings_cache = {}
    model = Code2Seq.load("java")
    path_to_reports = os.path.join(data_dir, FINAL_REPORTS_DIR)
    for file_name in iterate_reports(path_to_reports):
        report = Report.load_report(os.path.join(path_to_reports, file_name))
        commit_hash = report.hash
        for frame in report.frames:
            method_name = frame.meta["method_name"]
            method_key = (commit_hash, method_name)
            file_key = (commit_hash, frame.meta["path"])
            if file_key not in file_embeddings_cache:
                file_embeddings_cache[file_key] = get_file_embeddings(model, frame.get_code_decoded())
            if method_key not in embeddings:
                embeddings[method_key] = get_method_embedding(file_embeddings_cache[file_key], method_name)

    with open(os.path.join(data_dir, embs_name), 'w') as f:
        pickle.dump(embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--files_limit", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default='../data/')
    parser.add_argument("--embs_name", type=str, default='embs_dataset')

    args = parser.parse_args()
    get_all_embeddings(args.data_dir, args.save_dir, args.embs_name, args.files_limit)

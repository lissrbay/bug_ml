from pycode2seq import Code2Seq
import os
import argparse
from new.data.report import Report, Frame
from new.data.embs_dataset import EmbsDataset
import torch
from typing import Dict
from new.data_aggregation.utils import iterate_reports
from new.constants import CODE2SEQ_EMBEDDING_SIZE, CODE2SEQ_TMP_FILE, REPORTS_SUBDIR


def zeros() -> torch.Tensor:
    return torch.zeros(CODE2SEQ_EMBEDDING_SIZE)


def get_file_embeddings(model, filename: str) -> torch.Tensor:
    method_embeddings = model.methods_embeddings(filename)
    return method_embeddings


def get_method_embedding(method_embeddings, method_name: str) -> torch.Tensor:
    method_short_name = method_name.split('.')[-1]
    if method_short_name in method_embeddings:
        return method_embeddings[method_short_name]
    else:
        return zeros()


def code_to_tmp(code: str):
    with open(CODE2SEQ_TMP_FILE, 'w') as tmp_file:
        tmp_file.write(code)


def embed_frame(model, frame: Frame, cashed_preds: Dict) -> torch.Tensor:
    input_filename = frame.meta['file_name']
    if frame.code:
        if input_filename in cashed_preds:
            method_embeddings = cashed_preds[input_filename]
        else:
            code_to_tmp(frame.get_code_decoded())
            method_embeddings = get_file_embeddings(model, CODE2SEQ_TMP_FILE)
            cashed_preds[input_filename] = method_embeddings
        embedding = get_method_embedding(method_embeddings, frame.meta['method_name'])
    else:
        embedding = zeros()

    return embedding


def embed_frames(model, report, files_limit):
    cashed_preds = {}
    embeddings = []
    for frame in report.frames[:files_limit]:
        embedding = embed_frame(model, frame, cashed_preds)

        embeddings.append(embedding)
    return embeddings


def get_reports_embeddings(raw_reports_path: str, save_dir: str, embs_name: str, files_limit=80):
    embeddings = []
    report_ids = []
    model = Code2Seq.load("java")
    path_to_reports = os.path.join(raw_reports_path, REPORTS_SUBDIR)
    for file_name in iterate_reports(path_to_reports):
        path_to_file = os.path.join(path_to_reports, file_name)
        report = Report.load_report(path_to_file)
        report_embeddings = embed_frames(model, report, files_limit)
        report_ids.append(report.id)
        for i in range(len(report.frames), files_limit):
            report_embeddings.append(zeros())

        embeddings.extend(report_embeddings)
        break
    reports_count = len(report_ids)
    embs_dataset = EmbsDataset(report_ids,  torch.cat(embeddings).reshape(reports_count,
                                                                                 files_limit,
                                                                                 CODE2SEQ_EMBEDDING_SIZE))
    torch.save(embs_dataset,
               os.path.join(save_dir, embs_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reports_path", type=str)
    parser.add_argument("--files_limit", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default='../data/')
    parser.add_argument("--embs_name", type=str, default='embs_dataset')

    args = parser.parse_args()
    get_reports_embeddings(args.reports_path, args.save_dir, args.embs_name, args.files_limit)
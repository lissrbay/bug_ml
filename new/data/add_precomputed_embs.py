from report import Report
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embs_path", type=str)
    parser.add_argument("--report_ids_path", type=str)
    parser.add_argument("--path_to_reports", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


def add_embs_to_reports(report_ids_path, embs_path, path_to_reports, path_to_cached_reports):
    report_ids = np.load(report_ids_path)
    X = np.load(embs_path)
    for i in range(len(report_ids)):
        path_to_report = os.path.join(path_to_reports, report_ids[i])
        try:
            report = Report.load_report(path_to_report)
            embs_count = X[i].shape[0]
            for j in range(min(report.frames_count(), embs_count)):
                emb = X[i][j]
                report.frames[j].cached_embedding = emb
            new_path = os.path.join(path_to_cached_reports, report_ids[i])
            report.save_report(new_path)
        except Exception:
            print(f"Bad report: {path_to_report}")


if __name__ == "__main__":
    args = parse_args()

    add_embs_to_reports(args.report_ids_path, args.embs_path, args.path_to_reports, args.save_path)





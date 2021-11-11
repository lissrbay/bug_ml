from json.decoder import JSONDecodeError
from os import path
from pathlib import Path
from re import split
from typing import List, Optional, Tuple, Union
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torch.nn.functional import pad
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from model import DeepAnalyze

from modules import TfidfEmbeddings

class ReportsDataModule(pl.LightningDataModule):
    def __init__(self, path_to_reports, batch_size, embeddings):
        super().__init__()
        self.path_to_reports = path_to_reports
        self.embeddings = embeddings
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.rtrain, self.rval = train_test_split(ReportDataset(self.path_to_reports, self.embeddings), test_size=0.2, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rtrain, self.batch_size, collate_fn=pad_collate, num_workers=12)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.rval, self.batch_size, collate_fn=pad_collate, num_workers=12)


class ReportDataset(Dataset):
    def __init__(self, path_to_reports, embeddings, max_len=255) -> None:
        self.x, self.y = load_reports(path_to_reports, embeddings, max_len)
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.x[idx].shape[0]

def pad_collate(data):
    features, labels, lengths = zip(*data)
    max_len = max(lengths)
    features = [pad(feature, (0,0,0, max_len - length)).unsqueeze(1) for length, feature in zip(lengths, features)]
    labels = [pad(label, (0, max_len - length)).unsqueeze(1) for length, label in zip(lengths, labels)]

    features = torch.cat(features, dim=1)
    labels = torch.cat(labels, dim=1)

    mask = torch.arange(max_len).unsqueeze(-1).expand(max_len, len(lengths)) < torch.tensor(lengths).long().unsqueeze(0)

    return features.float(), labels.long(), mask


def load_reports(path_to_reports: Path, embeddings: TfidfEmbeddings, max_len):
    x, y = [], []
    for report in path_to_reports.glob("*.json"):
        try:
            with open(report, "r") as f:
                parsed_report = json.load(f)
            if parsed_report["frames"]:
                names = [frame["method_name"] for frame in parsed_report["frames"]][:max_len]
                frame_embeddings = embeddings.transform(names)
                label = torch.Tensor([frame["label"] for frame in parsed_report["frames"]][:max_len])
                x.append(frame_embeddings)
                y.append(label)
        except JSONDecodeError:
            print(report)
            continue
    return x, y
        

def build_embeddings(path_to_reports: Path) -> TfidfEmbeddings:
    namespace_docs = []
    method_docs = []

    for report in path_to_reports.glob("*.json"):
        try:
            with open(report, "r") as f:
                parsed_report = json.load(f)
            
            namespace_doc = []
            method_doc = []
            for frame in parsed_report["frames"]:
                tokens = frame["method_name"].split(".")
                method_doc.append(tokens[-1])
                namespace_doc.extend(tokens[:-1])

            namespace_docs.append(".".join(namespace_doc))
            method_docs.append(".".join(method_doc))
        except JSONDecodeError:
            print(report)
            continue

    embeddings = TfidfEmbeddings()

    embeddings.fit(method_docs, namespace_docs)

    return embeddings


def train(path_to_reports, embeddings_path: Path):
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings = torch.load(f)
    else:
        embeddings = build_embeddings(path_to_reports)
        with open(embeddings_path, "wb") as f:
            torch.save(embeddings, f)
    dm = ReportsDataModule(path_to_reports, 4, embeddings)
    model = DeepAnalyze(feature_size=embeddings.n_embed, lstm_hidden_size=200, lstm_num_layers=1, n_tags=2)

    trainer = Trainer(gpus=1)

    trainer.fit(model, dm)


if __name__ == "__main__":
    reports_path = "/home/dumtrii/Downloads/reports"
    embeddings_path = "emds"
    torch.autograd.set_detect_anomaly(True)
    train(Path(reports_path), Path(embeddings_path))


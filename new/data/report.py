import base64
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import attr
import pandas as pd

from new import data


@dataclass
class Code:
    begin: int
    end: int
    code: str


sys.modules['data'] = data

#
# @dataclass
# class Annotation:
#     file_name: str
#     commit_hash: str
#     # commit_hashes
#     # authors
#     timestamps: List[float]
#
#     @staticmethod
#     def from_dict(d: Dict[str, Any]) -> "Annotation":
#         return Annotation(
#             file_name=d["file_name"],
#             commit_hash=d["commit_hash"],
#             timestamps=d["timestamps"]
#         )
#
#     @staticmethod
#     def from_csv(path: Path) -> "Annotation":
#         commit_hash, file_name, _ = path.name.split(":")
#         df = pd.read_csv(path)
#         timestamps = df.timestamp.to_numpy().astype("int")
#         return Annotation(
#             file_name=file_name,
#             commit_hash=commit_hash,
#             timestamps=timestamps
#         )


@attr.s(frozen=False, auto_attribs=True)
class Frame:
    meta: Dict
    code: Code
    method_name: str
    file_name: str
    line: int
    path: str
    file_path: str
    has_recursion: bool
    label: int = attr.ib(0)

    # committed_date: int = attr.ib(0)
    # annotation: Optional[Annotation] = attr.ib(None)
    authored_date: int = attr.ib(0)
    author: str = attr.ib('no_author')

    def get_code_decoded(self):
        return base64.b64decode(self.code.code).decode("UTF-8")


@attr.s(frozen=True, auto_attribs=True)
class Report:
    id: int
    exceptions: str
    hash: str
    frames: List[Frame]
    timestamp: int

    @staticmethod
    def _read_frames_from_base(base_report: Dict):
        frames = []
        for frame in base_report['frames']:
            new_frame = Frame(
                code=Code(0, 0, ''),
                method_name=frame['method_name'],
                file_name=frame['file_name'] or '',
                line=frame['line_number'],
                path='',
                file_path=frame.get('file_path', '') or '',
                has_recursion=frame.get('has_recursion', False),
                label=frame.get('label', 0),
                # annotation=Annotation(),
                authored_date=0,
                author='no_author'
            )
            frames.append(new_frame)

        return frames

    @staticmethod
    def load_from_base_report(report_path: str):
        with open(report_path, 'r') as report_io:
            base_report = json.load(report_io)
        exceptions = base_report['class']
        report_id = base_report['id']
        commit_hash = ""
        if base_report['commit'] is not None:
            commit_hash = base_report['commit']['hash']
        frames = Report._read_frames_from_base(base_report)
        report = Report(report_id, exceptions, commit_hash, frames)

        return report

    @staticmethod
    def load_report(name: str):
        with open(name, 'rb') as report_io:
            return pickle.load(report_io)

    def save_report(self, name: str):
        with open(name, 'wb') as report_io:
            pickle.dump(self, report_io)

    def frames_count(self):
        return len(self.frames)

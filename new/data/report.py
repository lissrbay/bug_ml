import base64
import json
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict

import attr

from new import data


@dataclass
class Code:
    begin: int
    end: int
    code: str


sys.modules['data'] = data


@attr.s(frozen=True, auto_attribs=True)
class Frame:
    code: Code
    meta: Dict

    def get_code_decoded(self):
        return base64.b64decode(self.code.code).decode("UTF-8")


@attr.s(frozen=True, auto_attribs=True)
class Report:
    id: int
    exceptions: str
    hash: str
    frames: List[Frame]
    exception_time: int

    @staticmethod
    def _read_frames_from_base(base_report: Dict):
        frames = []
        for frame in base_report['frames']:
            method_meta = {
                'method_name': frame['method_name'],
                'file_name': '' if frame['file_name'] is None else frame['file_name'],
                'line': frame['line_number'],
                'path': '',
                'label': 0,
                'file_path': '',
                'has_recursion': 0 if not('has_recursion' in frame) else frame['has_recursion']
            }
            if 'label' in frame:
                method_meta['label'] = frame['label']

            if 'file_path' in frame and frame['file_path'] is not None:
                method_meta['file_path'] = frame['file_path']

            new_frame = Frame(Code(0, 0, ''), method_meta)
            frames.append(new_frame)

        return frames

    @staticmethod
    def load_from_base_report(report_path: str):
        with open(report_path, 'r') as report_io:
            base_report = json.load(report_io)
        exceptions = base_report['class']
        report_id = base_report['id']
        commit_hash = ""
        exception_time = int(base_report['timestamp'])
        if base_report['commit'] is not None:
            commit_hash = base_report['commit']['hash']
        frames = Report._read_frames_from_base(base_report)
        report = Report(report_id, exceptions, commit_hash, frames, exception_time)

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

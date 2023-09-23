import ast
import base64
import json
import pickle
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Code:
    begin: int
    end: int
    code: str

import attr


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
    exception_time: int = 0
    exception_hash: str = ''

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
        exception_time = int(base_report['timestamp'])
        commit_hash = base_report['hash']
        exception_hash = ""

        if base_report['commit'] is not None:
            exception_hash = base_report['commit']['hash']
        frames = Report._read_frames_from_base(base_report)
        report = Report(report_id, exceptions, commit_hash, frames, exception_time, exception_hash)

        return report

    @staticmethod
    def load_report(name: str):
        with open(name, 'rb') as report_io:
            return pickle.load(report_io)

    @staticmethod
    def load_report_from_json(path: Path):
        code_path = path.with_suffix('.code')

        with open(code_path) as f:
            code_json = json.load(f)

        frames = []
        for frame in code_json['stacktrace']:
            meta = {
                'method_name': frame['method_name'],
                'label': frame['scaffle_label'],
                'ground_truth': frame['label'],
                'ts': [anno['ts'] for anno in frame['annotations']],
                'author': [anno['author'] for anno in frame['annotations']],
                'line': frame.get('line', None)
            }

            frames.append(Frame(
                code=Code(0, 0, ast.literal_eval(frame['code']) if frame['code'] else b''),
                meta=meta
            ))

        with open(path) as f:
            report_json = json.load(f)

        return Report(
            id=report_json['id'],
            hash=report_json['hash'],
            frames=frames,
            exceptions=[],
            exception_hash=None,
            exception_time=report_json['ts']
        )

    def dump_to_code_and_stack(self):
        exceptions = self.exceptions
        exception_ts = 0
        hash = self.hash

        stacktrace = []
        new_frames = []
        for frame in self.frames:
            code = frame.code.code
            authors = list(frame.meta['authors']) if 'authors' in frame.meta else []
            ts = list([int(i) for i in frame.meta['ts']]) if 'ts' in frame.meta else []
            method_name = frame.meta['method_name']
            stacktrace.append(method_name)
            new_frame = {
                'method_name': method_name,
                'code': str(code),
                'scaffle_label': frame.meta['label'] if 'ground_truth' in frame.meta else -1.0,
                'label': frame.meta['ground_truth'] if 'ground_truth' in frame.meta else frame.meta['label'],
                'line': 0 if frame.meta['line'] is None else frame.meta['line'],
                'path_to_code': frame.meta['path'],
                'annotations': [{'ts': line_ts, 'author': line_author} for line_ts, line_author in zip(ts, authors)]
            }
            new_frames.append(new_frame)

        report_json = {'id': self.id,
                       'ts': exception_ts,
                       'exceptions': exceptions,
                       'hash': hash,
                       'stacktrace': stacktrace}
        code_json = {'id': int(self.id),
                     'stacktrace': new_frames}
        return code_json, report_json


    def save_report(self, name: str):
        with open(name, 'wb') as report_io:
            pickle.dump(self, report_io)

    def frames_count(self):
        return len(self.frames)

from typing import List, Dict
import json
import attr
import pickle
import base64


@attr.s(frozen=True, auto_attribs=True)
class Frame:
    code: str
    meta: Dict

    def get_code_decoded(self):
        return base64.b64decode(self.code).decode("UTF-8")

@attr.s(frozen=True, auto_attribs=True)
class Report:
    id: int
    exceptions: str
    hash: str
    frames: List[Frame]

    @staticmethod
    def _read_frames_from_base(base_report: Dict):
        frames = []
        for frame in base_report['frames']:
            method_meta = {'method_name': frame['method_name'],
                           'file_name': frame['file_name'],
                           'line': frame['line_number']}
            if 'label' in frame:
                method_meta['label'] = frame['label']

            if 'file_path' in frame:
                method_meta['file_path'] = frame['file_path']

            new_frame = Frame('', method_meta)
            frames.append(new_frame)

        return frames

    @staticmethod
    def load_from_base_report(name):
        with open(name, 'r') as report_io:
            base_report = json.load(report_io)
        exceptions = base_report['class']
        _id = base_report['id']
        frames = Report._read_frames_from_base(base_report)
        report = Report(_id, exceptions, '', frames)

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
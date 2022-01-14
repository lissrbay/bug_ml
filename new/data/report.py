from typing import List, Dict, Optional
import json
import attr
import pickle
import torch

@attr.s(frozen=False, auto_attribs=True)
class Frame:
    code: str
    meta: Dict
    cached_embedding: Optional[torch.Tensor]


@attr.s(frozen=False, auto_attribs=True)
class Report:
    id: int
    exceptions: str
    hash: str
    frames: List[Frame]

    def fill_frames(self, base_report: Dict):
        self.frames = []
        for frame in base_report['frames']:
            method_meta = {'method_name': frame['method_name'],
                           'file_name': frame['file_name'],
                           'line': frame['line_number'],
                           'exception_class': self.exceptions}

            new_frame = Frame('', method_meta, None)
            self.frames.append(new_frame)


    @staticmethod
    def load_from_base_report(name):
        try:
            f = open(name, 'r')
            base_report = json.load(f)
            f.close()
        except json.JSONDecodeError:
            return Report(0, [], '', [])

        exceptions = base_report['class']
        _id = base_report['id']
        report = Report(_id, exceptions, '', [])
        report.fill_frames(base_report)
        return report


    @staticmethod
    def load_report(name: str):
        return pickle.load(open(name, 'rb'))


    def save_report(self, name: str):
        f = open(name, 'wb')
        pickle.dump(self, f)
        f.close()

    def frames_count(self):
        return len(self.frames)
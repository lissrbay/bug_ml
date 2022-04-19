import dataclasses
import json
import os
from dataclasses import dataclass
from os.path import join
from typing import List, Dict, Any, Optional


@dataclass
class Frame:
    method_name: str
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    label: Optional[float] = None

    @staticmethod
    def from_jdict(jdict: Dict[str, Any]) -> "Frame":
        return Frame(method_name=jdict["method_name"],
                     file_name=jdict.get("file_name", None),
                     line_number=jdict.get("line_number", None),
                     label=float(jdict["label"]) if "label" in jdict else None)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Report:
    id: int
    timestamp: str
    exceptions: List[str]
    frames: List[Frame]

    @staticmethod
    def from_json(path: str) -> "Report":
        with open(path, "r") as file:
            report_jdict = json.load(file)

        frames = [Frame.from_jdict(frame_jdict) for frame_jdict in report_jdict["frames"]]
        return Report(id=int(report_jdict["id"]),
                      timestamp=report_jdict["timestamp"],
                      exceptions=report_jdict["class"],
                      frames=frames)

    def save_json(self, dir: str):
        os.makedirs(dir, exist_ok=True)

        report_dict = {
            "id": self.id,
            "timestamp": self.timestamp,
            "exceptions": self.exceptions,
            "frames": [frame.to_dict() for frame in self.frames]
        }

        path = join(dir, f"{self.id}.json")
        with open(path, "w") as file:
            json.dump(report_dict, file, indent=2)

    # @staticmethod
    # def _read_frames_from_base(base_report: Dict) -> List[Frame]:
    #     frames = []
    #     for frame in base_report['frames']:
    #         method_meta = {
    #             'method_name': frame['method_name'],
    #             'file_name': frame['file_name'],
    #             'line': frame['line_number'],
    #             'path': ''
    #         }
    #         if 'label' in frame:
    #             method_meta['label'] = frame['label']
    #
    #         if 'file_path' in frame:
    #             method_meta['file_path'] = frame['file_path']
    #
    #         new_frame = Frame('', method_meta)
    #         frames.append(new_frame)
    #
    #     return frames

    # @staticmethod
    # def load_from_base_report(report_path: str) -> "Report":
    #     with open(report_path, 'r') as report_io:
    #         base_report = json.load(report_io)
    #     exceptions = base_report['class']
    #     report_id = base_report['id']
    #     commit_hash = ""
    #     if base_report['commit'] is not None:
    #         commit_hash = base_report['commit']['hash']
    #     frames = Report._read_frames_from_base(base_report)
    #     report = Report(report_id, exceptions, commit_hash, frames)
    #
    #     return report

    # @staticmethod
    # def load_report(path: str) -> "Report":
    #     with open(path, 'rb') as report_io:
    #         return pickle.load(report_io)
    #
    # def save_report(self, path: str):
    #     with open(path, 'wb') as report_io:
    #         pickle.dump(self, report_io)
    #
    # def frames_count(self) -> int:
    #     return len(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

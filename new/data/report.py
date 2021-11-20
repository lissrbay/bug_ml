from typing import List

import attr


@attr.s(frozen=True, auto_attribs=True)
class Frame:
    name: str
    line: int
    code: str
    meta: str


@attr.s(frozen=True, auto_attribs=True)
class Report:
    id: int
    exceptions: str
    frames: List[Frame]

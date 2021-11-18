from typing import List

import attr


@attr.s(frozen=True, auto_attribs=True)
class Frame:
    name: str
    line: int


@attr.s(frozen=True, auto_attribs=True)
class Report:
    id: int
    exception: str
    frames: List[Frame]

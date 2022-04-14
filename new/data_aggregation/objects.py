from dataclasses import dataclass
from typing import Tuple


@dataclass
class Bounds:
    start_pos: int
    end_pos: int


@dataclass
class MethodSignature:
    name: str
    type: str


@dataclass
class Scope:
    name: str
    bounds: Bounds
    type: str

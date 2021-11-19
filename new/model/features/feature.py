from abc import ABC, abstractmethod
from typing import List

from new.data.report import Report


class BaseFeature(ABC):
    """
    It may be slow to compute every feature separately so we can group features and compute them together
    That's why we return a list of floats
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute(self, report: Report) -> List[List[float]]:
        raise NotImplementedError

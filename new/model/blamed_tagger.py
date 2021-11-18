from abc import ABC, abstractmethod
from typing import List

from new.data.report import Report


class BlamedTagger(ABC):
    @abstractmethod
    def predict(self, report: Report) -> List[float]:
        raise NotImplementedError

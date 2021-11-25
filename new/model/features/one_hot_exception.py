from collections import Counter
from typing import List

from new.data.report import Report


class OneHotExceptionClass:
    """
    One-hot features for most popular exception classes
    """
    def __init__(self, top_n: int = None):
        self.top_n = top_n
        self.top_exceptions = None

    def fit(self, reports: List[Report]) -> 'OneHotExceptionClass':
        exceptions = []
        for report in reports:
            exceptions += report.exceptions
        self.top_exceptions, _ = zip(*Counter(exceptions).most_common(self.top_n))
        return self

    def transform(self, reports: List[Report]) -> List[List[float]]:
        return [[1 if exception in report.exceptions else 0 for exception in self.top_exceptions] for report in reports]

    def fit_transform(self, df_features):
        self.fit(df_features)
        return self.transform(df_features)

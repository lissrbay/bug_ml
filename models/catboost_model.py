from collections import Counter
import pickle
from sklearn.model_selection import train_test_split

from catboost import CatBoost
from catboost import Pool

catboost_params = {'loss_function': 'QuerySoftMax',
                   'custom_metric': ['PrecisionAt:top=2', 'RecallAt:top=2', 'MAP:top=2'],
                   'eval_metric': 'AverageGain:top=2', 'depth': 3,  # eval_metric='AUC',
                   'metric_period': 100, 'iterations': 200}


class ExceptionTransformer:
    def __init__(self):
        self.exceptions = None

    def fit(self, df_features):
        exceptions_ = df_features['exception_class'].values
        exceptions = []
        for e in exceptions_:
            try:
                exceptions.append(e)
            except Exception:
                exceptions.append([])
        all_exceptions = []
        for exceptions_list in exceptions:
            for e in exceptions_list:
                all_exceptions.append(e)
        self.exceptions, _ = zip(*Counter(all_exceptions).most_common(100))

    def transform(self, df_features):
        df = df_features.copy()
        for e in self.exceptions:
            try:
                df[e] = df['exception_class'].apply(lambda x: 1 if e in x else 0)
            except Exception:
                print("No exception_class column.")
        return df

    def fit_transform(self, df_features):
        self.fit(df_features)
        return self.transform(df_features)


class CatBoostModel:
    def __init__(self):
        self.exception_transformer = ExceptionTransformer()
        self.model = None

    def train_test_splitting(self, df_features, fraction=0.9):
        df_features = self.exception_transformer.fit_transform(df_features)
        all_reports = df_features['report_id'].unique()
        reports_count = all_reports.shape[0]
        train_reports, test_reports = (
            all_reports[:int(reports_count * fraction)], all_reports[int(reports_count * fraction):])
        df_val = df_features[df_features['report_id'].isin(test_reports)].drop(
            ['method_name', 'indices', 'exception_class'], axis=1)

        df_features = df_features[df_features['report_id'].isin(train_reports)]
        X, test_X, y, test_y = train_test_split(
            df_features.drop(['label', 'method_name', 'indices', 'exception_class'], axis=1), df_features['label'],
            test_size=0.1, shuffle=False)
        train_dataset = Pool(X.drop(["report_id"], axis=1), y, group_id=X['report_id'])
        test_dataset = Pool(test_X.drop(["report_id"], axis=1), test_y, group_id=test_X['report_id'])
        return train_dataset, test_dataset, df_val

    def count_metrics(self, df_test):
        df_test['pred'] = self.model.predict(df_test.drop(['label'], axis=1))
        df_predicted = df_test.groupby("report_id").apply(lambda x: x.sort_values(ascending=False, by="pred").head(2))
        df_sum = df_predicted.reset_index(drop=True).groupby("report_id").sum()
        results = df_sum[df_sum["label"] >= 1].shape[0] / df_sum.shape[0]
        print(results)
        return results

    def train_catboost_model(self, train_pool, test_pool=None, save=False, path=''):
        self.model = CatBoost(catboost_params)
        self.model.fit(train_pool, eval_set=test_pool)
        if save:
            pickle.dump(self, open(path, "wb"))
        return self.model

    @classmethod
    def load_catboost_model(cls, path) -> 'CatBoostModel':
        return pickle.load(open(path, 'rb'))

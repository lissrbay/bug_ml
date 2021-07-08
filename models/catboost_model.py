
from numpy.lib.arraysetops import unique
from model import *
import os
import pandas as pd
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split

from catboost import CatBoost
from catboost import Pool


parser = argparse.ArgumentParser(description='Union features from sources')
parser.add_argument('data_path', type=str)
parser.add_argument('save_results', type=str)
parser.add_argument('save_model', type=str)


catboost_params = {'loss_function':'QuerySoftMax',
                    'custom_metric': ['PrecisionAt:top=2', 'RecallAt:top=2', 'MAP:top=2'],
                    'eval_metric': 'AverageGain:top=2','depth':3, #eval_metric='AUC', 
                    'metric_period':100, 'iterations':1000}


def train_test_split(df_features, fraction = 0.9):
    all_reports = df_features['report_id'].unique()
    reports_count = all_reports.shape[0]
    train_reports, test_reports = (all_reports[:int(reports_count*fraction)], all_reports[int(reports_count*fraction):])
    df_val = df_features[df_features['report_id'].isin(test_reports)]
    df_features = df_features[df_features['report_id'].isin(train_reports)]
    X, test_X, y, test_y = train_test_split(df_features.drop(['label', 'method_name', 'indices', 'exception_class', 'exception_class_'], axis=1), df_features['label'], 
                                        test_size=0.1, shuffle=False)
    train_dataset = Pool(X.drop(["report_id"], axis=1), y, group_id=X['report_id'])
    test_dataset = Pool(test_X.drop(["report_id"], axis=1), test_y, group_id=test_X['report_id'])
    return train_dataset, test_dataset, df_val


def count_metrics(model, df_test):
    test_X = df_test.drop(['label', 'method_name', 'report_id', 'indices'], axis=1)
    df_test['pred'] = model.predict(test_X) 
    df_predicted = df_test.groupby("report_id").apply(lambda x: x.sort_values(ascending=False, by="pred").head(2))
    df_sum = df_predicted.reset_index(drop=True).groupby("report_id").sum()
    results = df_sum[df_sum["label"] >= 1].shape[0]/df_sum.shape[0]
    
    return results


def train_catboost_model(train_pool, test_pool=None, save=False):
    model = CatBoost(catboost_params)
    model.fit(train_pool, eval_set=test_pool)
    if save:
        model.save_model(parser.save_model)
    return model

if __name__ == "__main__":
    df_features = pd.read_csv(parser.data_path)
    train_pool, test_pool, df_val = train_test_split(df_features)
    model = train_catboost_model(train_pool, test_pool, save=True)

    results = count_metrics(model, df_val)
    f = open(parser.save_results, 'w')
    json.dump(results, f, indent=4)
    f.close()



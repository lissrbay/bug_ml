import pandas as pd
import sys
import os
import pickle
import argparse
from collections import Counter


parser = argparse.ArgumentParser(description='Union features from sources')
parser.add_argument('data_path', type=str)
parser.add_argument('save_path', type=str)


def replace_exceptions(df_features):
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
    exceptions, _ = zip(*Counter(all_exceptions).most_common(100))
    df_features['exception_class_'] = df_features['exception_class']#.apply(eval)
    for e in exceptions:
        df_features[e] = df_features['exception_class_'].apply(lambda x: 1 if e in x else 0)
    return df_features


def make_indices(df_features):
    df_features['indices'] = df_features['report_id'].apply(lambda x: str(x)+'_') + df_features['method_stack_position'].apply(lambda x: str(x))
    return df_features


def union_preds_features(preds, df_features):
    df_features = replace_exceptions(df_features)
    df_features = make_indices(df_features)
    preds = make_indices(preds)
    df_features['indices'] = df_features['indices'].astype(str)
    preds['indices'] = preds['indices'].astype(str)
    preds = preds.drop(['report_id'], axis=1)
    print(df_features.shape, preds.shape)

    df_features = df_features.merge(preds, on='indices', how='inner')
    return df_features
    
    
if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_path,"lstm_prediction.csv"),index_col=0)
    df_features = pd.read_csv(os.path.join(args.data_path,"reports_features.csv"),index_col=0)
    reports_code = pickle.load(open(os.path.join(args.data_path, "reports_code"), "rb"))
    reports_ids = pickle.load(open(os.path.join(args.data_path,"reports_ids"), "rb"))

    df_features = replace_exceptions(df_features)
    df_features = make_indices(df_features)
    df = make_indices(df)

    df_features['indices'] = df_features['indices'].astype(str)
    df['indices'] = df['indices'].astype(str)  
    df_features = df_features.merge(df, on='indices', how='inner')

    df_features.to_csv(os.path.join(args.save_path))
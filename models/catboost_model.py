
from model import *
import os
import pandas as pd
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split

from catboost import CatBoost
from catboost import Pool

def replace_exceptions(df_features):
    exceptions = df_features['exception_class']
    exceptions = [eval(e) for e in exceptions]
    all_exceptions = []
    for exceptions_list in exceptions:
        for e in exceptions_list:
            all_exceptions.append(e)
    exceptions, _ = zip(*Counter(all_exceptions).most_common(100))
    df_features['exception_class_'] = df_features['exception_class'].apply(eval)
    for e in exceptions:
        df_features[e] = df_features['exception_class_'].apply(lambda x: 1 if e in x else 0)
    return df_features
def make_indices(df_features, reports_ids, reports_code):
    a = [[int(reports_ids[i])] * len(reports_code[i]) for i in range(len(reports_ids))]
    b = []
    for l in a:
        b.extend(l)
    df_features['report_id'] = b
    df_features['indices'] = df_features['report_id'].apply(lambda x: str(x)+'_') + df_features['method_stack_position'].apply(lambda x: str(x))
    return df_features

if __name__ == "__main__":
    path = os.path.join("..", "data")
    blm = BugLocalizationModel(os.path.join(path, "X.npy"), os.path.join(path, "y.npy"))
    params = blm.create_list_of_train_hyperparameters()
    blm.train(params, LSTMTagger)
    X, y, reports_used = read_data()
    preds = blm.model(FloatTensor(X))[:,:, 1]
    preds = preds.flatten()
    indices = np.array([[reports_used[j]+'_'+str(i) for i in range(80)] for j in range(len(reports_used))]).flatten()

    df = pd.DataFrame({"lstm":preds.detach().numpy(), "indices":indices})
    df.to_csv(os.path.join(path,"lstm_prediction.csv"))

    df= pd.read_csv(os.path.join(path,"lstm_prediction.csv"),index_col=0)
    df_features = pd.read_csv(os.path.join(path,"reports_features.csv"),index_col=0)
    reports_code=pickle.load(open(os.path.join(path, "reports_code"), "rb"))
    targets=pickle.load(open(os.path.join(path,"targets"), "rb"))
    reports_ids=pickle.load(open(os.path.join(path,"reports_ids"), "rb"))

    df_features = replace_exceptions(df_features)
    df_features = make_indices(df_features, reports_ids, reports_code)


    df_features['indices'] = df_features['indices'].astype(str)
    df['indices'] = df['indices'].astype(str)  
    df_features = df_features.merge(df, on='indices', how='inner')
    all_reports = df_features['report_id'].unique()
    train_reports, test_reports = all_reports[:int(all_reports.shape[0]*0.9)], all_reports[int(all_reports.shape[0]*0.9):]
    df_test = df_features[df_features['report_id'].isin(test_reports)]
    df_features = df_features[df_features['report_id'].isin(train_reports)]
    X, test_X, y, test_y = train_test_split(df_features.drop(['label', 'method_name', 'indices', 'exception_class', 'exception_class_'], axis=1), df_features['label'], 
                                        test_size=0.1, shuffle=False)
    model = CatBoost({'loss_function':'QuerySoftMax',
                    'custom_metric': ['PrecisionAt:top=2', 'RecallAt:top=2', 'MAP:top=2'],
                    'eval_metric': 'AverageGain:top=2','depth':3, #eval_metric='AUC', 
                    'metric_period':100, 'iterations':1000})
    categorical_indicies = [5]
    train_dataset = Pool(X.drop(["report_id"], axis=1), y, group_id=X['report_id'])
    test_dataset = Pool(test_X.drop(["report_id"], axis=1), test_y, group_id=test_X['report_id'])
    model.fit(train_dataset, eval_set=test_dataset)

    test_X = df_test.drop(['label', 'method_name', 'report_id', 'indices'], axis=1)
    test_y = df_test['label']
    (model.predict(test_X) == test_y)[model.predict(test_X)==1].sum()/np.sum(test_y)
    df_test['pred'] = model.predict(test_X) 
    df_predicted = df_test.groupby("report_id").apply(lambda x: x.sort_values(ascending=False, by="pred").head(2))
    df_sum = df_predicted.reset_index(drop=True).groupby("report_id").sum()#[df_predicted['label'] == df_predicted['pred']].shape[0] / df_predicted.shape[0]
    results = df_sum[df_sum["label"] >= 1].shape[0]/df_sum.shape[0]

    f = open('results_code2seq.txt', 'w')
    json.dump(results, f, indent=4)
    f.close()

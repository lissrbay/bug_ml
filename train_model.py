from models.model import *
import os
import torch.optim as optim
import data_aggregation.get_features
from models.catboost_model import train_test_splitting, train_catboost_model, count_metrics
import pickle
from data_aggregation.union_predictions_and_features import union_preds_features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reports_path", type=str, default='../intellij_fixed_201007')
parser.add_argument("--frame_limit", type=int, default=80)
class BugLocalizationModelTrain:
    def __init__(self, reports_path='', embeddings_path='./data/X(code2seq).npy', labels_path='./data/y(code2seq).npy',
                 report_ids_path='./data/reports_ids', report_code_path='./data/reports_code', frames_limit=80):
        self.model = None
        self.params = None
        self.path_to_embeddings = embeddings_path
        self.path_to_labels = labels_path
        self.path_to_report_ids = report_ids_path
        self.path_to_report_code = report_code_path
        self.path_to_reports = reports_path
        self.frames_limit = frames_limit

    def train_grid_search_model(self, save_path, params=None):
        blm = BugLocalizationModel(self.path_to_embeddings, self.path_to_labels)
        model = LSTMTagger
        if params is None:
            params = blm.create_list_of_train_hyperparameters()
        blm.train(params, model, top_two=True)
        self.params = params
        blm.save_results(name=save_path)
        self.model = blm


    def fit_model_from_params(self, params=None, use_best_params=False, path_to_results='.', save_path='./lstm_model'):
        blm = BugLocalizationModel(self.path_to_embeddings, self.path_to_labels)
        model = LSTMTagger

        if use_best_params:
            self.train_grid_search_model(save_path)
            best_params = self.model.best_params()
        else:
            best_params = params
        blm.train(best_params, model)
        blm.save_model(save_path)
        self.model = blm
        return blm

    def get_code_features(self):
        self.feature_extractor = data_aggregation.get_features.FeatureExtractor()
        self.feature_extractor.get_features_from_files(self.path_to_reports, self.path_to_report_code, self.path_to_report_ids)

        return self.feature_extractor.to_pandas()

    def get_lstm_train_preds(self):
        X = np.load(self.path_to_embeddings, allow_pickle=True)            
        preds = self.model.model(FloatTensor(X))[:,:, 1]
        return preds.flatten()

    def model_prediction_to_df(self, prediction):
        reports_ids=pickle.load(open(self.path_to_report_ids, "rb"))
        reports_ids_ = []
        poses = []
        for i in reports_ids:
            for j in range(self.frames_limit):
                reports_ids_.append(i)
                poses.append(j)
        return pd.DataFrame({'report_id':reports_ids_, 
        'method_stack_position': poses, 'lstm_prediction':prediction.detach().numpy()})

    def collect_data_for_catboost(self):
        code_features_df = self.get_code_features()
        lstm_prediction = self.get_lstm_train_preds()
        df_preds = self.model_prediction_to_df(lstm_prediction)
        df_all = union_preds_features(df_preds, code_features_df)

        return df_all

    def fit_catboost(self):
        df_features = self.collect_data_for_catboost()
        train_pool, test_pool, df_val = train_test_splitting(df_features)
        model = train_catboost_model(train_pool, test_pool, save=True, path='./cb_model')
        count_metrics(model, df_val)
        self.cb_model = model

        return model


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_report = os.path.join(args.reports_path, "labeled_reports")
    api = BugLocalizationModelTrain(reports_path=path_to_report)
    params = Parameters(0.01, 10, optim.Adam, 0.5, 5, 60)
    model = api.fit_model_from_params([params], save_path='./data/lstm')
    cb_model = api.fit_catboost()

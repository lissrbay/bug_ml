from models.model import BugLocalizationModel, Parameters, LitBugLocModel
from models.LSTMTagger import LSTMTagger
import os
import datetime
import torch.optim as optim
import data_aggregation.get_features
from models.catboost_model import CatBoostModel
import pickle
from data_aggregation.union_predictions_and_features import union_preds_features
import json
from models.dataset_wrapper import read_data, create_dataloader
import numpy as np
import pytorch_lightning as pl


class BugLocalizationModelTrain:
    def __init__(self, reports_path, embeddings_path, labels_path,
                 report_ids_path, report_code_path, frames_limit):
        self.model = None
        self.params = None
        self.path_to_embeddings = embeddings_path
        self.path_to_labels = labels_path
        self.path_to_report_ids = report_ids_path
        self.path_to_report_code = report_code_path
        self.path_to_reports = reports_path
        self.frames_limit = frames_limit
        self.postfix_hash = str(datetime.datetime.today().strftime("%Y%m%d_%H%M"))

    def train_grid_search_model(self, save_path, params=None):
        dataset = read_data(self.path_to_embeddings, self.path_to_labels)
        train_dataloader, test_dataloader = create_dataloader(dataset, test_size=0.1)
        embeddings_size = dataset[0][0].shape[1]
        #blm = BugLocalizationModel(embeddings_size=embeddings_size)
        model = LSTMTagger
        #if params is None:
        #    params = blm.create_list_of_train_hyperparameters()
        #blm.train(train_dataloader, test_dataloader, params, model, top_two=True)
        autoencoder = LitBugLocModel(model)

        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        # trainer = pl.Trainer(gpus=8) (if you have GPUs)
        trainer = pl.Trainer()
        trainer.fit(autoencoder, train_loader)
        self.params = params
        #blm.save_results(name=save_path)
        #self.model = blm


    def fit_model_from_params(self, params=None, use_best_params=False, path_to_results='.', save_dir=None, model_name=None):
        dataset = read_data(self.path_to_embeddings, self.path_to_labels)
        train_dataloader, test_dataloader = create_dataloader(dataset, test_size=0.1)
        embeddings_size = dataset[0][0].shape[1]
        #blm = BugLocalizationModel(embeddings_size=embeddings_size)
        model = LSTMTagger
        if save_dir is None:
            save_dir = '.'
        if model_name is None:
            model_name = 'lstm_' + self.postfix_hash
        save_path = os.path.join(save_dir, model_name)
        if use_best_params:
            self.train_grid_search_model(save_path)
            best_params = self.model.best_params()
        else:
            best_params = params
        autoencoder = LitBugLocModel(model)
        trainer = pl.Trainer()
        trainer.fit(autoencoder, train_dataloader)
        self.params = params
        #blm.train(train_dataloader, test_dataloader, best_params, model)

        #blm.save_model(save_path)
        #self.model = blm
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

    def fit_catboost(self, save_dir=None, model_name=None):
        df_features = self.collect_data_for_catboost()
        cbm = CatBoostModel()

        train_pool, test_pool, df_val = cbm.train_test_splitting(df_features)
        if save_dir == "" or save_dir is None:
            save_dir = '.'
        if model_name == "" or model_name is None:
            model_name = "cb_model_" + self.postfix_hash
        cbm.train_catboost_model(train_pool, test_pool, save=True, path=os.path.join(save_dir, model_name))
        cbm.count_metrics(df_val)
        self.cb_model = cbm.model

        return cbm.model


if __name__ == "__main__":
    args = json.load(open("train_properties.json", "r"))
    path_to_report = os.path.join(args["reports_path"], "labeled_reports")
    cur_dir = os.path.dirname(os. path. realpath(__file__))

    api = BugLocalizationModelTrain(reports_path=path_to_report,
                                    embeddings_path=os.path.join(cur_dir,args['embeddings_path']), 
                                    labels_path=os.path.join(cur_dir,args['labels_path']),
                                    report_ids_path=os.path.join(cur_dir,args['report_ids_path']), 
                                    report_code_path=os.path.join(cur_dir,args['report_code_path']),
                                    frames_limit=80)
    params = Parameters(0.01, 10, optim.Adam, 0.5, 5, 60)
    model = api.fit_model_from_params([params], save_dir=os.path.join(cur_dir, "" if args.get("save_dir") is None else args.get("save_dir")), model_name= args.get("lstm_model_name"))
    cb_model = api.fit_catboost(os.path.join(cur_dir, "" if args.get("save_dir") is None else args.get("save_dir")), model_name= args.get("cb_model_name"))

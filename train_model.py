from models.model import Parameters, PlBugLocModel
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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import FloatTensor
import torch
import pandas as pd 

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
        lrs = [1e-2, 1e-3]
        epochs = [20]
        anneal_coefs = [0.25]
        anneal_epochs = [5, 10, 15]
        hidden_dim = [40, 60, 80]
        optims = [optim.Adam]

    def create_list_of_train_hyperparameters(self):
        params = it.product(self.lrs, self.epochs, self.optims, self.anneal_coefs,
                            self.anneal_epochs, self.hidden_dim)
        params = map(lambda param: Parameters(*param), params)
        return params

    def train_grid_search_model(self, save_path, params=None):
        dataset = read_data(self.path_to_embeddings, self.path_to_labels)
        train_dataloader, test_dataloader = create_dataloader(dataset, test_size=0.1, shuffle=False)
        embeddings_size = dataset[0][0].shape[1]
        model = LSTMTagger

        self.run_records = []
        params = self.create_list_of_train_hyperparameters()
        for param in params:
            print(param)
            loss = self.criterion()
            self.model = model_to_train(
                word_emb_dim=self.embeddings_size,
                lstm_hidden_dim=param.dim)

            optimizer = param.optim(self.model.parameters(), lr=param.lr)
            
            logger = TensorBoardLogger("tb_logs", name="PlBugLocModel")

            autoencoder = PlBugLocModel(model, )
            trainer = pl.Trainer(logger=logger, log_every_n_steps=10)


        self.params = params


    def fit_model_from_params(self, params=None, use_best_params=False, path_to_results='.', save_dir=None, model_name=None):
        dataset = read_data(self.path_to_embeddings, self.path_to_labels)
        self.train_dataloader, self.test_dataloader = create_dataloader(dataset, test_size=0.2)
        embeddings_size = dataset[0][0].shape[1]
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
        logger = TensorBoardLogger("tb_logs", name="PlBugLocModel")

        autoencoder = PlBugLocModel(model)
        trainer = pl.Trainer(logger=logger, log_every_n_steps=10, max_epochs=20, callbacks=[ModelCheckpoint(dirpath=save_dir)])
        trainer.fit(autoencoder, self.train_dataloader, self.test_dataloader)
        self.model = autoencoder.model
        self.params = params
        autoencoder.save_model(save_path)
        return trainer

    def get_code_features(self):
        self.feature_extractor = data_aggregation.get_features.FeatureExtractor()
        self.feature_extractor.get_features_from_files(self.path_to_reports, self.path_to_report_code, self.path_to_report_ids)

        return self.feature_extractor.to_pandas()

    def get_lstm_train_preds(self):   
        X = np.load(self.path_to_embeddings, allow_pickle=True)
        X = torch.FloatTensor(X)
        preds = self.model(X)[:,:, 1]
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

    def save_results(self, name):
        results = []
        for i, record in enumerate(self.run_records):
            result = {'train_acc': record.train_history, 'val_acc': record.val_history}
            param = self.params[i]._asdict()
            param = dict(map(lambda item: (str(item[0]), str(item[1])), param.items()))
            result.update(param)
            results.append(result)
        results.sort(key=lambda x: -x['val_acc'])
        f = open('results' + name + '.txt', 'w')
        json.dump(results, f, indent=4)
        f.close()

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
    params = Parameters(0.001, 20, optim.Adam, 0.5, 5, 40)
    model = api.fit_model_from_params([params], save_dir=os.path.join(cur_dir, "" if args.get("save_dir") is None else args.get("save_dir")), model_name= args.get("lstm_model_name"))
    cb_model = api.fit_catboost(os.path.join(cur_dir, "" if args.get("save_dir") is None else args.get("save_dir")), model_name= args.get("cb_model_name"))

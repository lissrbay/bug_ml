import base64
import json

import numpy as np
import pandas as pd
import torch
from pycode2seq import Code2Seq

import data_aggregation.get_features
from data_aggregation.union_predictions_and_features import union_preds_features
from models.catboost_model import CatBoostModel
from models.model import BugLocalizationModel


class BugLocalizationModelAPI:
    def __init__(self, emb_model, lstm_model_path='', cb_model_path='', frames_limit=384):
        self.emb_model = emb_model
        self.model = None
        if lstm_model_path:
            self.model = BugLocalizationModel()
            self.model.load_model(lstm_model_path)

        if cb_model_path:
            self.cb_model = CatBoostModel.load_catboost_model(cb_model_path)

        self.code2seq_predictor = None
        self.frames_limit = 384

    def get_code_features(self, methods_data):
        self.feature_extractor = data_aggregation.get_features.FeatureExtractor()
        for method in methods_data:
            self.feature_extractor.get_feature_from_code(method['code'])
            self.feature_extractor.get_feature_from_metadata(method['meta'])
        return self.feature_extractor.to_pandas()

    def collect_data_for_catboost(self, methods_data, lstm_prediction):
        code_features_df = self.get_code_features(methods_data)
        df_preds = self.model_prediction_to_df(lstm_prediction, methods_data)
        df_all = union_preds_features(df_preds, code_features_df)
        df_all = df_all.drop(['label', 'method_name', 'report_id', 'indices'], axis=1)
        return df_all

    def predict_bug_lstm(self, embeddings, top_k=3):
        prediction = self.model.model(torch.FloatTensor(embeddings))[:, :, 1]
        prediction = prediction.flatten()
        return (-prediction).argsort()[:top_k].tolist(), prediction.tolist()

    def predict_bug_cb(self, catboost_data, top_k=3):
        prediction = self.cb_model.predict_proba(catboost_data)[:, :, 1]
        prediction = prediction.flatten()
        return (-prediction).argsort()[:top_k], prediction

    @staticmethod
    def model_prediction_to_df(prediction, methods_data):
        return pd.DataFrame(
            {'report_id': [frame['meta']['id'] for frame in methods_data], 'method_stack_position': np.arange(0, len(prediction)),
             'lstm_prediction': prediction})

    def predict(self, methods_data, pred_type='lstm', top_k=3):
        embeddings = self.get_embeddings(methods_data)
        top_k_pred, lstm_prediction = self.predict_bug_lstm(embeddings, top_k)
        if pred_type == 'lstm':
            return top_k_pred, lstm_prediction

        catboost_data = self.collect_data_for_catboost(methods_data, lstm_prediction)

        if pred_type == 'all':
            return self.predict_bug_cb(catboost_data, top_k)

    def get_embeddings(self, methods_data):
        embeddings = []
        for method in methods_data:
            code = base64.b64decode(method['code']).decode("UTF-8")
            if code != "":
                f = open("tmp.java", "w")
                f.write(code)
                f.close()
                method_embeddings = self.emb_model.methods_embeddings("tmp.java")
                method_short_name = method['meta']['method_name'].split('.')[-1]
                if method_short_name in method_embeddings:
                    embeddings.append(method_embeddings[method_short_name].detach().numpy())
                else:
                    embeddings.append(np.zeros(320))
            else:
                embeddings.append(np.zeros(320))
        return np.array(embeddings).reshape(1, -1, 320)


def main():
    emb_model = Code2Seq.load("java")
    stacktrace = json.load(open("ex_api_stacktrace.json", "r"))
    api = BugLocalizationModelAPI(emb_model,
                                  lstm_model_path='./data/lstm_20210909_1513',
                                  cb_model_path='./data/cb_model_20210909_1513')
    top_k_pred, scores = api.predict(stacktrace, pred_type='all')

    print(top_k_pred)
    print(scores)


if __name__ == '__main__':
    main()

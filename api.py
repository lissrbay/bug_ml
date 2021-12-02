import base64
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from pycode2seq import Code2Seq

from data_aggregation.get_features import FeatureExtractor
from data_aggregation.union_predictions_and_features import union_preds_features
from models.LSTMTagger import LSTMTagger
from models.catboost_model import CatBoostModel
from models.model import PlBugLocModel


class BugLocalizationModelAPI:
    _tmp_file_name = "tmp.java"
    _emb_dim = 320

    def __init__(self, emb_model, lstm_model_path: str, cb_model_path: str = '', frames_limit: int = 384):
        self.emb_model = emb_model
        pl_model = PlBugLocModel(LSTMTagger)
        pl_model.load_model(lstm_model_path)
        self.model = pl_model

        if cb_model_path:
            self.cb_model = CatBoostModel.load_catboost_model(cb_model_path)
        self.frames_limit = frames_limit

    @staticmethod
    def get_code_features(methods_data: List[Dict[str, Any]]) -> pd.DataFrame:
        feature_extractor = FeatureExtractor()
        for method in methods_data:
            feature_extractor.get_feature_from_code(method['code'])
            feature_extractor.get_feature_from_metadata(method['meta'])
        df_features = feature_extractor.to_pandas()
        return df_features

    def predict_bug_lstm(self, embeddings: np.ndarray, top_k: int = 3) -> Tuple[List[int], List[float]]:
        prediction = self.model.model(torch.FloatTensor(embeddings))[:, :, 1]
        prediction = prediction.flatten()
        return (-prediction).argsort()[:top_k].tolist(), prediction

    def predict_bug_cb(self, methods_data: List[Dict[str, Any]], lstm_prediction: List[float],
                       top_k: int = 3) -> Tuple[List[int], List[float]]:
        code_features_df = self.get_code_features(methods_data)
        report_id = methods_data[0]['meta']['id'] if 'id' in methods_data[0]['meta'] else 0
        df_preds = self.model_prediction_to_df(report_id, lstm_prediction)
        df_all = union_preds_features(df_preds, code_features_df)
        df_all = df_all.drop(['label', 'method_name', 'report_id', 'indices'], axis=1)
        prediction = self.cb_model.predict(df_all)
        return (-prediction).argsort()[:top_k].tolist(), prediction.tolist()  # type: ignore

    @staticmethod
    def model_prediction_to_df(report_id, prediction):
        return pd.DataFrame({'report_id': [report_id for _ in range(len(prediction))],
                             'method_stack_position': np.arange(0, len(prediction)),
                             'lstm_prediction': prediction})

    def predict(self, methods_data: List[Dict[str, Any]], pred_type: str = 'lstm', top_k: int = 3):
        embeddings = self.get_embeddings(methods_data)
        top_k_pred, lstm_prediction = self.predict_bug_lstm(embeddings, top_k)
        if pred_type == 'lstm':
            return top_k_pred, lstm_prediction

        if pred_type == 'all':
            return self.predict_bug_cb(methods_data, lstm_prediction, top_k)

    def get_embeddings(self, methods_data: List[Dict[str, Any]]):
        embeddings = []
        for method in methods_data:
            code = base64.b64decode(method['code']).decode("UTF-8")
            if code != "":
                with open(BugLocalizationModelAPI._tmp_file_name, "w") as f:
                    f.write(code)
                # TODO: return full method name
                method_embeddings = self.emb_model.methods_embeddings(BugLocalizationModelAPI._tmp_file_name)
                method_short_name = method['meta']['method_name'].split('.')[-1]
                if method_short_name in method_embeddings:
                    embeddings.append(method_embeddings[method_short_name].detach().numpy())
                else:
                    embeddings.append(np.zeros(BugLocalizationModelAPI._emb_dim))
            else:
                embeddings.append(np.zeros(BugLocalizationModelAPI._emb_dim))
        return np.array(embeddings).reshape((1, -1, BugLocalizationModelAPI._emb_dim))


def main():
    emb_model = Code2Seq.load("java")
    with open("ex_api_stacktrace.json", "r") as f:
        stacktrace = json.load(f)
    api = BugLocalizationModelAPI(emb_model,
                                  lstm_model_path='./data/lstm_20211024_2053',
                                  cb_model_path='./data/cb_model_20211024_2053')
    top_k_pred, scores = api.predict(stacktrace, pred_type='all')

    print(top_k_pred)
    print(scores)


if __name__ == '__main__':
    main()

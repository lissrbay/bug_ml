from models.model import *
from code2seq.code2seq import predict as predict_embeddings
import data_aggregation.get_features
from models.catboost_model import load_catboost_model
from embeddings.match_embeddings_with_methods import match_embeddings_with_methods_from_df
from data_aggregation.union_predictions_and_features import union_preds_features
import json
import base64

class BugLocalizationModelAPI:
    def __init__(self, lstm_model_path='', cb_model_path='', frames_limit=384):
        self.model = None
        if lstm_model_path:
            self.model = BugLocalizationModel()
            self.model.load_model(lstm_model_path)

        if cb_model_path:
            self.cb_model = load_catboost_model(cb_model_path)

        self.code2seq_predictor = None
        self.frames_limit = 384


    def get_code_features(self, methods_data):
        self.feature_extractor = data_aggregation.get_features.FeatureExtractor()
        for method in methods_data:
            self.feature_extractor.get_feature_from_code(method['code'])
            self.feature_extractor.get_feature_from_code(method['meta'])
        return self.feature_extractor.to_pandas()

    def collect_data_for_catboost(self):
        code_features_df = self.get_code_features()
        lstm_prediction = self.get_lstm_train_preds()
        df_preds = self.model_prediction_to_df(lstm_prediction)
        df_all = union_preds_features(df_preds, code_features_df)

        return df_all

    def predict_bug_lstm(self, embeddings, top_k=3):
        prediction = self.model.model(FloatTensor(embeddings))[:,:, 1]
        prediction = prediction.flatten()
        return (-prediction).argsort()[:top_k], prediction

    def predict_bug_cb(self, catboost_data, top_k=3):
        prediction = self.cb_model.predict_proba(catboost_data)[:,:, 1]
        prediction = prediction.flatten()
        return (-prediction).argsort()[:top_k], prediction
        
    def model_prediction_to_df(self, prediction, frames_len):
        return pd.DataFrame({'report_id':np.zeros(frames_len), 'method_stack_position': np.arange(0,frames_len), 'lstm_prediction':prediction})

    def predict(self, methods_data, pred_type='lstm', top_k=3):
        embeddings = self.get_embeddings(methods_data)
        top_k_pred, lstm_prediction = self.predict_bug_lstm(embeddings, top_k)
        if pred_type == 'lstm':
            return top_k_pred, lstm_prediction

        catboost_data = self.collect_data_for_catboost(methods_data, lstm_prediction)
        
        if pred_type == 'all':
            return self.predict_bug_cb(catboost_data, top_k)


    def get_embeddings(self, methods_data):
        if self.code2seq_predictor is None:
            self.code2seq_predictor = predict_embeddings()
        methods_embeddings = []
        for method in methods_data:
            embeddings_df = None
            if method['code']:
                embeddings_df = self.code2seq_predictor.predict_by_code(base64.b64decode(method['code']).decode("UTF-8"))
            embedding = match_embeddings_with_methods_from_df(embeddings_df, method['meta'])
            methods_embeddings.append(embedding)
        frames_len = len(methods_data)

        for _ in range(frames_len, self.frames_limit):
            methods_embeddings.append(np.zeros(384))

        return np.array(methods_embeddings)



from models.model import *
import data_aggregation.get_features
from models.catboost_model import load_catboost_model
from embeddings.match_embeddings_with_methods import match_embeddings_with_methods_from_df
from data_aggregation.union_predictions_and_features import union_preds_features
import json
import base64
from pycode2seq import Code2Seq


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
            self.feature_extractor.get_feature_from_metadata(method['meta'])
        return self.feature_extractor.to_pandas()

    def collect_data_for_catboost(self, methods_data, lstm_prediction):
        code_features_df = self.get_code_features(methods_data)
        frames_len = len(methods_data)
        df_preds = self.model_prediction_to_df(lstm_prediction, frames_len)
        df_all = union_preds_features(df_preds, code_features_df)
        df_all = df_all.drop(['label', 'method_name', 'report_id', 'indices'], axis=1)
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
        prediction = prediction.detach().numpy()
        return pd.DataFrame({'report_id':np.zeros(prediction.shape[0]), 'method_stack_position': np.arange(0,prediction.shape[0]), 'lstm_prediction':prediction})

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
                method_embeddings = model.methods_embeddings("tmp.java")
                method_short_name = method['meta']['method_name'].split('.')[-1]
                if method_short_name in method_embeddings:
                    embeddings.append(method_embeddings[method_short_name].detach().numpy())
            else:
                embeddings.append(np.zeros(320))
        return np.array(embeddings).reshape(1, -1, 320)

if "__name__" == "__main__":
    model = Code2Seq.load("java")
    method_embeddings = model.methods_embeddings("Line.java")

    stacktrace = json.load(open("ex_api_stacktrace.json", "r"))
    api = BugLocalizationModelAPI(lstm_model_path='./data/lstm', cb_model_path='./data/cb_model')
    top_k_pred, _ = api.predict(stacktrace, pred_type='lstm')
    print(top_k_pred)
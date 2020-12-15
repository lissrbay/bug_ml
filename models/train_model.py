from model import *
import sys
from flat_model import SimpleModel


model_types = ['code2seq', 'code2vec']
code2vec_paths = {
    'embeddings_path': 'X(code2vec).npy',
    'labels_path': 'y(code2vec).npy',
    'reports_path': 'report_ids(code2vec).npy'
}
code2vec_wv_paths = {
    'embeddings_path': 'X(wv).npy',
    'labels_path': 'y(wv).npy',
    'reports_path': 'report_ids(code2vec).npy'
}

def train_model(model, paths=None, name='', ranked=False, flat=False):
    if paths:
        blm = BugLocalizationModel(**paths, ranked=ranked, flat=flat)
    else:
        blm = BugLocalizationModel(ranked=ranked, flat=flat)

    params = blm.create_list_of_train_hyperparameters()
    blm.train(params, model, top_one=False)
    name = name if not ranked else name+'_ranked'
    blm.save_results(name=name)


if len(sys.argv) > 1:
    model_type = sys.argv[1]
ranked = False
flat = False
model = LSTMTagger

if len(sys.argv) > 2:
    if sys.argv[2] == 'ranked':
        ranked = True
    if sys.argv[2] == 'flat':
        flat = True
        model = SimpleModel

if model_type == 'code2seq':
    train_model(model, name='code2seq', ranked=ranked, flat=flat)
elif model_type == 'code2vec':
    train_model(model, code2vec_paths, name='code2vec', ranked=ranked, flat=flat)
elif model_type == 'code2vec(wv)':
    train_model(model, code2vec_wv_paths, name='code2vec', ranked=ranked, flat=flat)

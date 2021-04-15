from model import *
import sys
from flat_model import SimpleModel
import os

model_types = ['code2seq', 'code2vec']
code2vec_paths = {
    'embeddings_path': os.path.join("..", "data", 'X(code2vec).npy'),
    'labels_path': os.path.join("..", "data",'y(code2vec).npy'),
    'reports_path': os.path.join("..", "data",'report_ids(code2vec).npy')
}
code2vec_wv_paths = {
    'embeddings_path': os.path.join("..", "data",'X(wv).npy'),
    'labels_path': os.path.join("..", "data",'y(wv).npy'),
    'reports_path': os.path.join("..", "data",'report_ids(code2vec).npy')
}
code2seq_paths = {
    'embeddings_path': os.path.join("..", "data",'X.npy'),
    'labels_path': os.path.join("..", "data",'y.npy'),
    'reports_path': os.path.join("..", "data",'report_ids.npy')
}
def train_model(model, paths=None, name='', ranked=False, flat=False, top_two_accuracy=False):
    if paths:
        blm = BugLocalizationModel(**paths, ranked=ranked, flat=flat)
    else:
        blm = BugLocalizationModel(ranked=ranked, flat=flat)

    params = blm.create_list_of_train_hyperparameters()
    blm.train(params, model, top_two=top_two_accuracy)
    name = name if not ranked else name+'_ranked'
    name = name if not flat else name+'_flat'
    name = name if not top_two_accuracy else name+'top_two'

    blm.save_results(name=name)

model_type = 'code2seq'
if len(sys.argv) > 1:
    model_type = sys.argv[1]
ranked = False
flat = False
top_two_accuracy = False
model = LSTMTagger
params = sys.argv[1:]

for param in params:
    if param == 'ranked':
        ranked = True
    if param == 'flat':
        flat = True
        model = SimpleModel
    if param == 'top_two':
        top_two_accuracy = True

if model_type == 'code2seq':
    train_model(model, name='code2seq', ranked=ranked, flat=flat, top_two_accuracy=top_two_accuracy)
elif model_type == 'code2vec':
    train_model(model, code2vec_paths, name='code2vec', ranked=ranked, flat=flat, top_two_accuracy=top_two_accuracy)
elif model_type == 'code2vec_wv':
    train_model(model, code2vec_wv_paths, name='code2vec_wv', ranked=ranked, flat=flat, top_two_accuracy=top_two_accuracy)
elif model_type == 'all':
    for name in ['code2seq', 'code2vec', 'code2vec_wv']:
        for ranked in [True, False]:       
            train_model(model, name=name, ranked=ranked, flat=False, top_two_accuracy=False)
        for flat  in [True, False]:       
            train_model(model, name=name, ranked=False, flat=flat, top_two_accuracy=False)
        for top_two_accuracy in [True, False]:
            train_model(model, name=name, ranked=False, flat=False, top_two_accuracy=top_two_accuracy)

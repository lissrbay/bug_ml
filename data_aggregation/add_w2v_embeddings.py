import gensim.models as gm
import re
import numpy as np
from tqdm import tqdm
import json
import sys


def find_closest_vector(model, method_name, embeddings_size=384):
    word_vectors = []
    for word in re.findall('[A-Z][^A-Z]*', method_name):
        try:
            closest_word = model.most_similar(positive=[word.lower()])[0][0]
            word_vectors.append(model[closest_word])
        except Exception:
            word_vectors.append(np.zeros(embeddings_size))
    if not word_vectors:
        word_vectors.append(np.zeros(embeddings_size))

    return np.mean(word_vectors)


def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    return method_name


def find_name_embeddings(X, model, reports_used, embed_all_methods = False, max_frames=80):
    path_to_report = '/home/lissrbay/Рабочий стол/code2vec/code2vec_experiments/code2seq/labeled_reports'
    for n, report_id in tqdm(enumerate(reports_used)):
        f = open(path_to_report + '/' + report_id + '.json', 'r')
        report = json.load(f)
        f.close()
        for i, frame in enumerate(report['frames'][:max_frames]):
            method_name = clean_method_name(frame['method_name'])
            if i < X.shape[1] and X[n][i].sum() == 0.0:
                X[n][i] = find_closest_vector(model, method_name)
            elif embed_all_methods:
                X[n][i] = find_closest_vector(model, method_name)
    return X


model = gm.KeyedVectors.load_word2vec_format('target_vecs.txt', binary=False)
X = np.load('X(code2vec).npy')
y = np.load('y(code2vec).npy')
reports_used = np.load('report_ids(code2vec).npy')
if len(sys.argv) > 1:
    X = find_name_embeddings(X, model, reports_used, embed_all_methods=True)
    np.save('X(wv_all)', X)
    np.save('y(wv_all)', y)
else:
    X = find_name_embeddings(X, model, reports_used)
    np.save('X(wv)', X)
    np.save('y(wv)', y)
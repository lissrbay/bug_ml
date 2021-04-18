from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from keras.models import Sequential, Model
from keras.layers import Dense, InputLayer, Input, concatenate, Flatten, Lambda
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


def clean_method_name(method_name):
    method_name = method_name.split('.')[-1]
    method_name = method_name.replace('lambda$', '')
    method_name = method_name.replace('$0', '')
    method_name = method_name.replace('$', '')
    method_name = method_name.replace('_', '')
    method_name = method_name.replace('<', '')
    method_name = method_name.replace('>', '')

    return method_name


def collect_texts(path_to_report):
    max_features = 1000
    ngram_max = 1
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    tfidf_model = tfidf.fit(texts_)

    target = []
    texts = []
    for n, report_id in tqdm(enumerate(reports_used)):
        f = open(path_to_report + '/' + report_id + '.json', 'r')
        report = json.load(f)
        f.close()
        for i, frame in enumerate(report['frames']):
            if i < X.shape[1] and (X[n][i] ** 2).sum() != 0.0:
                method_name = clean_method_name(frame['method_name'])
                method_name = ' '.join(list(re.findall('[A-Z][^A-Z]*', method_name)))
                v = method_name
                if not ('run' == method_name):
                    target.append(X[n][i])
                    texts.append(v)
    texts = tfidf_model.transform(texts).toarray()
    return texts


def baseline_model(input_dim):
    tfidf_input = Input(shape=(input_dim,))
    x = Dense(units=400, kernel_initializer='normal', activation='relu')(tfidf_input)
    out = Dense(units=320, kernel_initializer='normal', activation='linear')(x)
    model = Model(inputs=[tfidf_input], outputs=out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def make_rec(vec, vecs, vecs_norm, vocab, K=20):
    scores = np.matmul(vecs, vec.reshape(320))

    norm = np.linalg.norm(vec)
    if norm != 0:
        scores /= (vecs_norm * norm)

    top_K = np.argpartition(scores, -K)[-K:]
    top_scores = scores[top_K]
    top_K = top_K[np.argsort(top_scores)[::-1]]

    rec = [vocab[vec_index] for vec_index in top_K]
    scores = [float(s) for s in np.sort(top_scores)[::-1]]

    return list(zip(rec, scores))


if __name__ == "__main__":
    path_to_report = '/home/lissrbay/Рабочий стол/code2vec/code2vec_experiments/code2seq/labeled_reports'
    texts_  = collect_texts(path_to_report)
    max_features = 1000
    ngram_max = 1
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    tfidf_model = tfidf.fit(texts_)
    texts = tfidf_model.transform(texts).toarray()
    X_train, X_test, Y_train, Y_test = train_test_split(texts, target)
    model = baseline_model(1000)

    model_path = os.path.join('./', 'baseline_model')
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(X_train, Y_train, epochs=2, batch_size=64, callbacks=[checkpoint],
    validation_data=(X_test, Y_test))
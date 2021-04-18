import pickle
import os
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from model import *

import numpy as np
from tqdm import tqdm
def add_bert_embeddings(model, reports_code, targets, reports_ids):
    X = []
    y = []
    for i in tqdm(range(len(reports_code))[:5]):
        X.append([])
        y.append([])
        for j in range(80):
            if j >= len(reports_code[i]):
                vec = np.zeros((768,))
                y[i].append(0)
            else:
                code_tokens=tokenizer.tokenize(reports_code[0][0].replace("\n", '')[:256])
                tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
                with torch.no_grad():
                    context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
                    vec = context_embeddings.mean(axis=1).reshape((768,))
                del tokens_ids, code_tokens
                y[i].append(targets[i][j])
            X[i].append(vec)
    X_ = np.zeros((4000, 80, 768))
    for i in range(4000):
        for j in range(80):
            for k in range(768):
            X_[i][j][k] = X[i][j][k]
    X = np.array(X_)
    return X, np.array(y)


if __name__ == "__main__":
    path = os.path.join("..", "data")

    reports_code=pickle.load(open(os.path.join(path, "reports_code"), "rb"))
    targets=pickle.load(open(os.path.join(path,"targets"), "rb"))
    reports_ids=pickle.load(open(os.path.join(path,"reports_ids"), "rb"))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    X, y = add_bert_embeddings(model, reports_code, targets, reports_ids)
    pickle.dump(X, open(os.path.join(path,"X_robert"), "wb"))
    pickle.dump(y, open(os.path.join(path,"y_robert"), "wb"))
    blm = BugLocalizationModel(open(os.path.join(path,"X_robert"), open(os.path.join(path,"y_robert"))
    params = blm.create_list_of_train_hyperparameters()
    blm.train(params, LSTMTagger)




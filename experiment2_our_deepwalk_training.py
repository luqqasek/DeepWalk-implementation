from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import numpy as np
from helpers import read_from_pyg
import time
from DeepWalkOur import DeepWalkOurs
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import networkx as nx
warnings.filterwarnings("ignore")

# GLOBAL PARAMETERS FOR EXPERIMENT
########################################################################################################################
# Path to file with graph
path = 'inj_cora.pt'
# Name of the file to output results
output_file_name = "gensim_grid_search_nohs_massive.csv"
# number of re runs of model
number_of_reruns = 5
# Parameters of models to train
models_param = {"nohs_lr_bin": {"hs": False,
                                "dimensions": 128,
                                "walk_number": 1,
                                "walk_length": 2,
                                "window_size": 2},
                "nohs_lr_mac": {"hs": False,
                                "dimensions": 512,
                                "walk_number": 1,
                                "walk_length": 8,
                                "window_size": 32},
                "hs_xgb_bin_mac": {"hs": True,
                                   "dimensions": 256,
                                   "walk_number": 16,
                                   "walk_length": 32,
                                   "window_size": 32},
                "our_params_nohs": {"hs": False,
                                    "dimensions": 500,
                                    "walk_number": 4,
                                    "walk_length": 70,
                                    "window_size": 2},
                "our_params_hs": {"hs": True,
                                  "dimensions": 500,
                                  "walk_number": 4,
                                  "walk_length": 70,
                                  "window_size": 2}
                }
########################################################################################################################


def predict(G, embeddings, return_f1_score=False):
    y = list(nx.get_node_attributes(G, 'y').values())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, shuffle=True, stratify=y)

    # Train logistic regression
    model_ = LogisticRegression(class_weight='balanced')
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)

    pos_samples = sum(y)
    neg_samples = len(y) - pos_samples
    model_ = XGBClassifier(max_depth=10, scale_pos_weight=neg_samples / pos_samples)

    model_.fit(X_train, y_train)
    y_pred_xgb = model_.predict(X_test)

    # Evaluate predictions
    if return_f1_score:
        a_ = f1_score(y_test, y_pred, average="binary")
        b_ = f1_score(y_test, y_pred, average="macro")
        c_ = f1_score(y_test, y_pred_xgb, average="binary")
        d_ = f1_score(y_test, y_pred_xgb, average="macro")
        cm1_ = confusion_matrix(y_test, y_pred, normalize='true')
        cm2_ = confusion_matrix(y_test, y_pred_xgb, normalize='true')
        return a_, b_, c_, d_, cm1_, cm2_
    else:
        print(f1_score(y_test, y_pred, average="binary"))
        print(f1_score(y_test, y_pred, average="binary"))
        print(confusion_matrix(y_test, y_pred, normalize='true'))


if __name__ == "__main__":
    G = read_from_pyg(path)

    f = open("our_trained_on_best.csv", "a")
    f.write(
        "name,dimensions,number_of_walks,walk_length,window_size,duration,f1_binary_lr,f1_macro_lr,f1_binary_xgb,f1_macro_xgb,duration_sd,f1_binary_lr_sd,f1_macro_lr_sd,f1_binary_xgb_sd,f1_macro_xgb_sd\n")
    f.close()

    for model in models_param:
        params = models_param[model]

        bf1_lr = []
        mf1_lr = []
        bf1_xgb = []
        mf1_xgb = []
        du = []

        window = params["window_size"]
        dims = params["dimensions"]
        num_walks = params["walk_number"]
        walk_length = params["walk_length"]

        for k in range(number_of_reruns):

            start_time = time.time()
            deepwalk = DeepWalkOurs(G=G, window=window, dims=dims, num_walks=num_walks, walk_length=walk_length,
                                    use_hierarchical_softmax=params["hs"])
            deepwalk.deepwalk()
            end_time = time.time()

            a, b, c, d, e, f = predict(G, deepwalk.embedder.input_layer.weight.data.cpu().numpy(), True)
            bf1_lr.append(a)
            mf1_lr.append(b)
            bf1_xgb.append(c)
            mf1_xgb.append(d)
            du.append(end_time - start_time)

        f = open("our_trained_on_best.csv", "a")
        f.write(
            f"{model},{dims},{num_walks},{walk_length},{window},{np.mean(d)},{np.mean(bf1_lr)},{np.mean(mf1_lr)},{np.mean(bf1_xgb)},{np.mean(mf1_xgb)},{np.std(d)},{np.std(bf1_lr)},{np.std(mf1_lr)},{np.std(bf1_xgb)},{np.std(mf1_xgb)}\n")
        f.close()

        print(model)
        print(f"Duration : {round(np.mean(du), 3)} +/- {round(np.std(du), 3)}")
        print(f"Binary F1 logistic regression : {round(np.mean(bf1_lr), 3)} +/- {round(np.std(bf1_lr), 3)}")
        print(f"Macro F1 logistic regression : {round(np.mean(mf1_lr), 3)} +/- {round(np.std(mf1_lr), 3)}")
        print(f"Binary F1 xgboost : {round(np.mean(bf1_xgb), 3)} +/- {round(np.std(bf1_xgb), 3)}")
        print(f"Macro F1 xgboost : {round(np.mean(mf1_xgb), 3)} +/- {round(np.std(mf1_xgb), 3)}")
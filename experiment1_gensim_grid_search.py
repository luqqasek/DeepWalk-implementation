from DeepWalkGensim import DeepWalkGensim
from helpers import *
from tqdm import tqdm
import itertools
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# GLOBAL PARAMETERS FOR EXPERIMENT
########################################################################################################################
dimensions = [32, 64, 128, 256, 512]
num_walks = [1, 4, 8, 16, 32]
walk_length = [2, 4, 8, 16, 32]
windws = [2, 4, 8, 16, 32]
# Path to file with graph
path = 'inj_cora.pt'
# Name of the file to output results
output_file_name = "gensim_grid_search_nohs_massive.csv"
# number of re runs of model
number_of_reruns = 5
########################################################################################################################

if __name__ == "__main__":
    G = read_from_pyg(path)

    f = open(output_file_name, "a")
    f.write("dimensions,number_of_walks,walk_length,window_size,duration,f1_binary_lr,f1_macro_lr,f1_binary_xgb,f1_macro_xgb,duration_sd,f1_binary_lr_sd,f1_macro_lr_sd,f1_binary_xgb_sd,f1_macro_xgb_sd\n")
    f.close()

    combinations = list(itertools.product(windws, dimensions, num_walks, walk_length))

    for comb in tqdm(combinations):
        window = comb[0]
        dims = comb[1]
        num_walks = comb[2]
        walk_length = comb[3]

        bf1_lr = []
        mf1_lr = []
        bf1_xgb = []
        mf1_xgb = []
        d = []

        for i in range(number_of_reruns):
            start = time.time()
            deepwalk = DeepWalkGensim(G, window=window, dims=dims, num_walks=num_walks, walk_length=walk_length)
            walks = deepwalk.generate_walks()
            embeddings = deepwalk.skipgram()
            end = time.time()
            duration = end - start
            binary_f1_lr, macro_f1_lr = predict(G, embeddings, return_f1_score=True, model='lr')
            binary_f1_xgb, macro_f1_xgb = predict(G, embeddings, return_f1_score=True, model='xgb')

            bf1_lr.append(binary_f1_lr)
            mf1_lr.append(macro_f1_lr)
            bf1_xgb.append(binary_f1_xgb)
            mf1_xgb.append(macro_f1_xgb)
            d.append(duration)

        f = open(output_file_name, "a")
        f.write(
            f"{dims},{num_walks},{walk_length},{window},{np.mean(d)},{np.mean(bf1_lr)},{np.mean(mf1_lr)},{np.mean(bf1_xgb)},{np.mean(mf1_xgb)},{np.std(d)},{np.std(bf1_lr)},{np.std(mf1_lr)},{np.std(bf1_xgb)},{np.std(mf1_xgb)}\n")
        f.close()

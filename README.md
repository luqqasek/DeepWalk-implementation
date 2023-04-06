# DeepWalk-implementation

This repository consists of DeepWalk implementations one using gensim word2vec model and second entirely in pytorch. Two implementations are then compared on task of outlier detection on dataset inj_cora coming from https://github.com/pygod-team/data

## Files description

#### Class and function definitions

[DeepWalkGensim](DeepWalkGensim.py) - consists of DeepWalkGensim class implementation that performs DeepWalk using gensim skipgram implementation

[DeepWalkOur](DeepWalkOur.py) - consists of DeepWalkOurs class implementation that performs DeepWalk using our implementation of Huffman tree, Hierarchical Softmax and SkipGram model

[PairsDataset](PairsDataset.py) - consists of PairsDataset class implementation that is a torch Dataset class that accepts list of pairs of (center node, context node)

[HierarchicalSoftmax](HierarchicalSoftmax.py) - consists of HierarchicalSoftmax layer implementation

[HuffmanTree](HuffmanTree.py) - consists of HuffmanNode and HuffmanTree class implementation

[SkipGram](SkipGram.py) - consists of NeuralEmbedder class that is SkipGram model implementation with HierarchicalSoftmax class if specified by user

[helpers](helpers.py) - consists of helpers functions definitions. read_from_pyg(...) read .pt graph file, predict(...) given embeddings predicts outliers using XGBoost or LogisticRegression classifier on 20% test train split, create_freq_dict(...) creates dictionary of number of visits in each node while doing a walk of given lenght from each node. 

#### Scripts

[experiment1_gensim_grid_search](experiment1_gensim_grid_search.py) - script that performs grid search on DeepWalk with gensim SkipGram implementation  

[experiment2_our_deepwalk_training](experiment2_our_deepwalk_training.py) - script that performs grid search on DeepWalk with our SkipGram implementation

#### Python notebook files

[Graphs2.ipynb](csvs_with_results/Graphs2.ipynb) - summarization of experiment 1 and creation of sensitivity graphs

[Benchmarking.ipynb](Benchmarking.ipynb) - benchmark of other models 


from torch.utils.data import DataLoader
import numpy as np
from PairsDataset import PairsDataset
from helpers import create_freq_dict
from SkipGram import NeuralEmbedder
import torch
import networkx as nx
import random
from tqdm import tqdm


class DeepWalkOurs:
    """
    G: the graph of which the node embeddings will be created
    window: the window size
    dims: the embedding size
    num_walks: The number of random walks performed for each individual node
    walk_length: The random walk length
    use_hierarchical_softmax: boolean value to specify whether to use or not hierarchical softmax
    use_cross_entropy: boolean value to specify whether to use or not cross entropy
    """

    def __init__(self, G: nx.Graph, window: int, dims: int, num_walks: int, walk_length: int,
                 use_hierarchical_softmax=False, use_cross_entropy=False, disable_walk_progress_bar=False,
                 lr_decay=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr_decay = lr_decay
        self.device = device
        self.G = G
        self.window_size = window
        self.dims = dims
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walks = None
        self.vocab = list(set(G.nodes))
        self.vocab_size = len(self.vocab)
        # dict of node name --> index
        self.vocab_to_idx = {self.vocab[i]: i for i in range(self.vocab_size)}
        # dict of index --> node name
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        if use_hierarchical_softmax:
            freq_dict = create_freq_dict(G, self.window_size)
            total_count = sum(freq_dict.values())
            freq_dict = {key: value / total_count for key, value in freq_dict.items()}
            self.embedder = NeuralEmbedder(node_count=self.vocab_size,
                                           embedding_dims=dims,
                                           use_hierarchical_softmax=True,
                                           freq_dict=freq_dict).to(device)
        else:
            self.embedder = NeuralEmbedder(node_count=self.vocab_size,
                                           embedding_dims=dims,
                                           use_cross_entropy=use_cross_entropy).to(device)
        self.visited_nodes_idx = set()  # according to paper lr is decreased based on number of visited nodes
        self.disable_walk_progress_bar = disable_walk_progress_bar
        # PARAMETER FOR TRAINING
        self.initial_lr = 0.025
        self.minimum_lr = 0.0001
        self.number_of_epochs = 1
        self.batch_size = 32  # can i do it more than once at a time - to think
        self.optimizer = torch.optim.SGD(self.embedder.parameters(), lr=self.initial_lr)
        self.all_losses = []

    def deepwalk(self):
        """
        The full deepwalk algorithm to be called by the user outside of this class
        """
        # store all vertices
        self.walks = []
        for _ in range(self.num_walks):
            if not self.disable_walk_progress_bar:
                print(f"\nWalk number {_+1}")
            # each iteration here makes a pass over the data to sample one walk for each node
            # the random order of the vertices speeds up the convergence of SGD
            random.shuffle(self.vocab)
            for v in tqdm(self.vocab, disable=self.disable_walk_progress_bar):
                # yields a generator object
                random_walk = self.RandomWalk(self.vocab_to_idx[v])
                # save walk
                self.walks.append(random_walk)
                # encode walk
                self.SkipGram(random_walk)
                # update visited nodes
                self.visited_nodes_idx = self.visited_nodes_idx.union(set(random_walk))

    def RandomWalk(self, v_idx):
        # v is the root
        walk = [v_idx]
        current_idx = v_idx
        for _ in range(self.walk_length):
            # neighbors of the current node
            current = self.idx_to_vocab[current_idx]
            neighbors = [node for node in self.G.neighbors(current)]
            # choose a neighbor to go to
            current_idx = self.vocab_to_idx[np.random.choice(neighbors)]
            walk.append(current_idx)

        return walk

    def SkipGram(self, walk):
        """
        walks: all the generated random walks which act as the equivalent of sentences.
        d: the embedding size
        w: the window size
        """
        # creates pairs to pass through SkipGram
        pairs = self.create_pairs_from_walk(walk)
        # creates dataset and dataloader for Skip Gram
        walk_dataset = PairsDataset(pairs)
        dataloader = DataLoader(walk_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.number_of_epochs):
            if self.lr_decay:
                # it maybe should happen in different place
                self.update_lr()

            for batch in dataloader:
                # Getting vectors and labels from batch
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Predicting
                self.optimizer.zero_grad()
                loss = self.embedder(inputs, targets)
                # Backward propagation
                loss.backward()
                self.optimizer.step()

    def create_pairs_from_walk(self, walk):
        """For given walk creates all pairs to pass through SkipGram"""
        pairs = []
        for center_word_pos in range(len(walk)):
            for w in range(-self.window_size, self.window_size + 1):
                # if window size is 2, go to -2,-1,0,1,2
                context_word_pos = center_word_pos + w
                if context_word_pos < 0 or context_word_pos >= len(walk) or context_word_pos == center_word_pos:
                    continue
                pairs.append((walk[center_word_pos], walk[context_word_pos]))
        return pairs

    def update_lr(self):
        # lr is decreased based on visited nodes
        visited_perc = len(self.visited_nodes_idx)/self.vocab_size
        # compute new learning rate
        new_lr = self.initial_lr * (1 - visited_perc) + visited_perc * self.minimum_lr
        # change learning rate
        self.optimizer.param_groups[0]['lr'] = new_lr

    def reset_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.initial_lr

import networkx as nx
from gensim.models import Word2Vec
import random
import numpy as np


class DeepWalkGensim:
    """
    Using this class: after initializing, run the generate_walks method to generate the random walks.
    To generate the embeddings, run the skipgram method.
    """

    def __init__(self,
                 G: nx.Graph,
                 window: int,
                 dims: int,
                 num_walks: int,
                 walk_length: int,
                 use_hs=0,
                 workers_number=24):
        """
        G: the graph of which the node embeddings will be created
        window: the window size
        dims: the embedding size
        num_walks: The number of random walks performed for each individual node
        walk_length: The random walk length
        """
        self.G = G
        self.window = window
        self.dims = dims
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walks = None
        self.use_hs = use_hs
        self.workers = workers_number

    def generate_walks(self):
        """
        Generating the walks
        """
        # store all vertices
        V = list(self.G.nodes)
        self.walks = []
        for _ in range(self.num_walks):
            # each iteration here makes a pass over the data to sample one walk for each node
            # the random order of the vertices speeds up the convergence of SGD
            random.shuffle(V)
            for v in V:
                # yields a generator object
                self.walks.append(self.RandomWalk(v))

        return self.walks

    def RandomWalk(self, v):
        """
        Generating one random walk
        """
        # v is the root
        walk = [v]
        current = v
        for _ in range(self.walk_length):
            # neighbors of the current node
            neighbors = [node for node in self.G.neighbors(current)]
            # choose a neighbor to go to
            current = np.random.choice(neighbors)
            walk.append(current)

        return walk

    def skipgram(self):
        """
        Calling the Word2Vec model that we will implement ourselves
        """

        model = Word2Vec(sentences=self.walks, vector_size=self.dims, window=self.window, negative=0, sg=1, hs=0,
                         workers=self.workers, min_count=1)

        # return the embeddings (word vectors)
        return model.wv

from torch import nn
from HierarchicalSoftmax import HierarchicalSoftmaxLayer


class NeuralEmbedder(nn.Module):
    '''
    node_count: equivalent to the "size of vocab"
    embedding_dims: output embedding dimensions / NN hidden layer weight matrix dimension
    '''

    def __init__(self, node_count, embedding_dims, use_hierarchical_softmax=False, use_cross_entropy=False,
                 freq_dict=None):
        super(NeuralEmbedder, self).__init__()
        self.node_count = node_count
        self.embedding_dims = embedding_dims
        self.use_hierarchical_softmax = use_hierarchical_softmax
        self.use_cross_entropy = use_cross_entropy
        self.input_layer = nn.Embedding(self.node_count,
                                        self.embedding_dims)
        # weights initialization
        nn.init.xavier_uniform_(self.input_layer.weight.data)

        # adding classification layer
        if use_hierarchical_softmax:
            assert freq_dict, "Provide frequency dictionary for building Huffman tree"
            # hierarchical softmax
            self.hs = HierarchicalSoftmaxLayer(self.node_count, self.embedding_dims, freq_dict)
        else:
            self.output_layer = nn.Linear(in_features=self.embedding_dims,
                                          out_features=self.node_count)

    def forward(self, center, target):
        # Embedding matrix computation
        embds = self.input_layer(center.view(-1))
        # Prediction of next word
        if self.use_hierarchical_softmax:
            loss = self.hs(embds, target.long())
        elif self.use_cross_entropy:
            output = self.output_layer(embds)
            loss = nn.functional.F.cross_entropy(output, target)
        else:
            output = self.output_layer(embds)
            log_probs = nn.functional.F.log_softmax(output, dim=1)
            loss = nn.functional.F.nll_loss(log_probs, target.view(-1), reduction='mean')
        return loss
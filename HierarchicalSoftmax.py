import torch
from torch import nn
from HuffmanTree import *
import torch.nn.functional as F


class HierarchicalSoftmaxLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, freq_dict):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_matrix = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=vocab_size)

        nn.init.xavier_uniform_(self.output_matrix.weight.data)
        self.huffman_tree = HuffmanTree(freq_dict)

    def forward(self, input_word, target):
        # input_word: [b_size, embedding_dim]
        # target: [b_size, 1]

        # turns:[b_size, max_code_len_in_batch]
        # paths: [b_size, max_code_len_in_batch]
        turns, paths = self._get_turns_and_paths(target)
        paths_emb = self.output_matrix(paths)  # [b_size, max_code_len_in_batch, embedding_dim]

        # storing dot product of input word and node representation
        dot_products = ((paths_emb * input_word.unsqueeze(1)).sum(2) * turns)
        # calculating loss using fact that sum of log is product of their argument
        loss = -F.logsigmoid(dot_products).sum(1).mean()

        return loss

    def _get_turns_and_paths(self, target):
        turns = []  # turn right(1) or turn left(-1) in huffman tree
        paths = []
        max_len = 0

        for n in target:
            n = n.item()
            node = self.huffman_tree.node_dict[n]

            code = target.new_tensor(node.code).int()  # in code, left node is 0; right node is 1
            turn = torch.where(code == 1, code, -torch.ones_like(code))

            turns.append(turn)
            paths.append(target.new_tensor(node.node_path))

            if node.code_len > max_len:
                max_len = node.code_len

        turns = [F.pad(t, pad=(0, max_len - len(t)), mode='constant', value=0) for t in turns]
        paths = [F.pad(p, pad=(0, max_len - p.shape[0]), mode='constant', value=self.vocab_size) for p in paths]
        return torch.stack(turns).int(), torch.stack(paths).long()

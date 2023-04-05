
class HuffmanNode:
    def __init__(self, is_leaf, value=None, fre=0, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value  # the node's index in huffman tree
        self.fre = fre  # word frequency in corpus
        self.code = []  # huffman code
        self.code_len = 0  # length of code
        self.node_path = []  # the path from root node to this node
        self.left = left  # left child
        self.right = right  # right child

    def __str__(self):
        return f"is_leaf : {self.is_leaf}\nvalue : {self.value}\nfrequency : {self.fre}\ncode : {self.code}\ncode_length : {self.code_len}\nnode_path : {self.node_path}"


class HuffmanTree:
    def __init__(self, fre_dict):
        self.root = None
        freq_dict = sorted(fre_dict.items(), key=lambda x: x[1], reverse=True)
        self.vocab_size = len(freq_dict)
        self.node_dict = {}
        self._build_tree(freq_dict)

    def _build_tree(self, freq_dict):
        """ building huffman tree and updating nodes information"""

        node_list = [HuffmanNode(is_leaf=True, value=w, fre=fre) for w, fre in freq_dict]  # create leaf node
        node_list += [HuffmanNode(is_leaf=False, fre=1e10) for i in range(self.vocab_size)]  # create non-leaf node

        parentNode = [0] * (self.vocab_size * 2)  # only 2 * vocab_size - 2 be used
        binary = [0] * (self.vocab_size * 2)  # recording turning left or turning right

        # pos1 points to currently processing leaf node at left side of node_list
        # pos2 points to currently processing non-leaf node at right side of node_list

        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size

        # building tree
        for a in range(self.vocab_size - 1):
            # first pick assigns to min1i
            if pos1 >= 0:
                if node_list[pos1].fre < node_list[pos2].fre:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1

            # second pick assigns to min2i
            if pos1 >= 0:
                if node_list[pos1].fre < node_list[pos2].fre:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1

            # fill information of non leaf node
            node_list[self.vocab_size + a].fre = node_list[min1i].fre + node_list[min2i].fre
            node_list[self.vocab_size + a].left = node_list[min1i]
            node_list[self.vocab_size + a].right = node_list[min2i]

            # assign lead child (min2i) and right child (min1i) to parent node
            parentNode[min1i] = self.vocab_size + a
            parentNode[min2i] = self.vocab_size + a
            binary[min2i] = 1

        # generate huffman code
        for a in range(self.vocab_size):
            b = a
            code = []
            point = []

            # going up until root
            while b != self.vocab_size * 2 - 2:
                code.append(binary[b])
                b = parentNode[b]
                point.append(b)

            # reversing since huffman code is from top to bottom
            node_list[a].code_len = len(code)
            node_list[a].code = list(reversed(code))

            node_list[a].node_path = list(reversed([p - self.vocab_size for p in point]))
            self.node_dict[node_list[a].value] = node_list[a]

        self.root = node_list[2 * self.vocab_size - 2]

from Node import Node, InputNode


class Layer:
    def __init__(self, numOfNodes):
        self.numOfNodes = numOfNodes
        self.nodes = [Node() for _ in range(numOfNodes)]

class InputLayer(Layer):
    def __init__(self, numOfNodes):
        super().__init__(numOfNodes)
        self.nodes = [InputNode() for _ in range(numOfNodes)]

    


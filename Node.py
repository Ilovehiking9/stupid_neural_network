from Activation import leakyRelu
import random
from math import sqrt

class Node:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.previousLayer = None
        self.output = 0
        self.error = 0

    def init(self, previousLayer):
        self.previousLayer = previousLayer
        self.weights = [random.gauss(0, sqrt(1/3)) for _ in previousLayer.nodes]
        self.biases = [random.uniform(-0.1, 0.1) for _ in previousLayer.nodes]  # Small random biases        
        
    def adjustWeight(self, weightIndex, adjustment):
        self.nodeInputs[weightIndex] = adjustment
        ...

    def setWeight(self, weightIndex, num):
        self.nodeInputs[weightIndex] += num
        ...
    
    def adjustBias(self, biasIndex, adjustment):
        self.nodeInputs[biasIndex] += adjustment
        ...
    
    def setBias(self, biasIndex, num):
        self.nodeInputs[biasIndex] = num
        ...

    def getOutput(self):
        if self.previousLayer is None:
            raise ValueError("Previous layer is not set. Please initialize the node with a previous layer.")
        
        sum = 0

        for i, node in enumerate(self.previousLayer.nodes):
            sum += node.getOutput() * self.weights[i] + self.biases[i]
            
        self.output = leakyRelu(sum)
        return self.output
    
    def calculatedError(self, expectedOutput=None, nextLayer=None):
        if expectedOutput is not None and nextLayer is None:
            # For output nodes
            self.error = self.output - expectedOutput  # Derivative of MSE
        elif expectedOutput is None and nextLayer is not None:
            # For hidden nodes
            self.error = 0
            for i, next_node in enumerate(nextLayer.nodes):
                self.error += next_node.error * next_node.weights[i]  # Weighted sum of next layer errors
            
            # Multiply by the derivative of Leaky ReLU
            self.error *= 1 if self.output > 0 else 0.01  # Leaky ReLU derivative
        else:
            print("Something has gone very wrong")
            return None
        
    def updateWeightsAndBiases(self, learning_rate):
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.error * self.previousLayer.nodes[i].output
        
        # Update biases
        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * self.error

class InputNode(Node):
    def __init__(self):
        super().__init__()
        self.inputValue = 0

    def setInput(self, value):
        self.inputValue = value

    def getOutput(self, previousLayerOutputs=None):
        return self.inputValue
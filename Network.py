from Layer import Layer, InputLayer
import json

class Network:
    def __init__(self):
        self.layers = []

    def addInputLayer(self, numOfNodes):
        self.layers.append(InputLayer(numOfNodes))
    def addLayer(self, numOfNodes):
        self.layers.append(Layer(numOfNodes))
 
    #initializes the network with random weights and biases. will reset everything!!
    def init(self):
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.init(self.layers[self.layers.index(layer) - 1])
        

    def printNetwork(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {layer.numOfNodes} nodes")

        print("\nWeights and Biases:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}:")
            for j, node in enumerate(layer.nodes):
                print(f"  Node {j + 1}:")
                print(f"    Weights: {node.weights}")
                print(f"    Biases: {node.biases}")

    def forward(self, inputNodeInputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.setInput(inputNodeInputs[i])

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].nodes)):
                self.layers[i].nodes[j].getOutput()

    def backprop(self, expectedOutputs, learning_rate):
        # Calculate errors for output layer
        for i, node in enumerate(self.layers[-1].nodes):
            node.calculatedError(expectedOutput=expectedOutputs[i])
        
        # Propagate errors backward through hidden layers
        for layer_index in range(len(self.layers) - 2, 0, -1):  # Skip input and output layers
            for node in self.layers[layer_index].nodes:
                node.calculatedError(nextLayer=self.layers[layer_index + 1])
        
        # Update weights and biases for all layers except the input layer
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.updateWeightsAndBiases(learning_rate)

    def calculateAverageCost(self, dataset):
        """
        Calculate the average cost over the entire dataset.
        :param dataset: List of tuples [(inputs, expected_outputs), ...]
        :return: Average cost (float)
        """
        total_cost = 0
        for inputs, expected_outputs in dataset:
            # Forward pass
            self.forward(inputs)
            
            # Calculate cost for this sample
            for i, node in enumerate(self.layers[-1].nodes):
                total_cost += 0.5 * (node.output - expected_outputs[i]) ** 2  # MSE
        
        # Return the average cost
        print (total_cost, len(dataset))
        return total_cost / len(dataset)

    def saveWeightsAndBiases(self, filepath):
        """
        Save the weights and biases of the network to a JSON file.
        """
        data = []
        for layer in self.layers[1:]:  # Skip the input layer
            layer_data = []
            for node in layer.nodes:
                node_data = {
                    "weights": node.weights,
                    "biases": node.biases
                }
                layer_data.append(node_data)
            data.append(layer_data)
        
        with open(filepath, 'w') as file:
            json.dump(data, file)
        print(f"Weights and biases saved to {filepath}")

    def loadWeightsAndBiases(self, filepath):
        """
        Load the weights and biases of the network from a JSON file.
        """
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        for layer, layer_data in zip(self.layers[1:], data):  # Skip the input layer
            for node, node_data in zip(layer.nodes, layer_data):
                node.weights = node_data["weights"]
                node.biases = node_data["biases"]
        print(f"Weights and biases loaded from {filepath}")


if __name__ == "__main__":
    from loadData import loadDataset

    network = Network()

    network.addInputLayer(6)
    network.addLayer(8)
    network.addLayer(4)
    network.addLayer(2)


    dataset = loadDataset("data.csv")
    network.init()

    network.loadWeightsAndBiases("weights_and_biases.json")


    # Training parameters
    learning_rate = 0.01
    epochs = 10000

    # Training loop
    for epoch in range(epochs):
        for inputs, expected_outputs in dataset:
            network.forward(inputs)
            network.backprop(expected_outputs, learning_rate)
        
        average_cost = network.calculateAverageCost(dataset)
        print(f"Epoch {epoch + 1}, Average Cost: {average_cost}")

        # Save weights and biases every 10 epochs
        if (epoch + 1) % 10 == 0:
            network.saveWeightsAndBiases("weights_and_biases.json")

    # Save final weights and biases
    network.saveWeightsAndBiases("weights_and_biases.json")

    # Load weights and biases for testing
    
import json
import numpy as np
from activation_functions import ActivationFunction
from layer import Layer


class NeuralNet:
    def __init__(self):
        # The neural network is a collection of layers, which is initialized as an empty list of layers
        self.layers = []

    # Add a layer to the neural network
    def add_layer(self, layer):
        self.layers.append(layer)

    # Propogate the input data through the ENTIRE neural network
    def forward_propagation(self, input_data):
        # If there are no layers, then we can't do forward propagation
        if len(self.layers) == 0:
            raise ValueError("No layers found in the neural network.")

        # We initialize the output variable to be equal to the input data, which is the input vector to the neural network
        # This variable will be updated as the data passes through each layer during forward propagation.
        output = input_data

        # We then loops through each layer in the neural network's layers
        # For each layer, we call the forward propagation method of that layer, passing the current output as the input data
        for layer in self.layers:
            output = layer.compute_propogation(output)

        return output
    
    
    # Save the neural network weights and biases to a text file
    def save(self, filename):
        # Create an empty dictionary to store weights, biases, and other layer information
        weights_and_biases = {}

        # Loop through each layer in the neural network's layers
        for i, layer in enumerate(self.layers):
            # Generate a layer_name for each layer based on its index in the list (0-based)
            layer_name = f"Layer_{i}"

            # Store the layer's weights and biases in the dictionary as Python lists
            # The `tolist()` method is used to convert NumPy arrays to regular Python lists for JSON serialization
            weights_and_biases[layer_name] = {
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),

                # Store additional information about layer and previous layer sizes
                "previousLayer_size": layer.previousLayer_size,
                "layer_size": layer.layer_size,

                # Store the title of the activation function as a string for later reference
                "activation_func": str(layer.activation_func.title),

                # Store the layer type ('input', 'hidden', or 'output') as a string
                "layer_type": layer.layer_type,
            }

        # Open the file in write mode and save the weights and biases dictionary as JSON
        with open(filename, "w") as file:
            json.dump(weights_and_biases, file)


    # Load the neural network weights and biases from a text file
    def load(self, filename):
        # Open the JSON file in read mode and load its contents into the 'data' variable
        with open(filename, "r") as file:
            data = json.load(file)

        # Initialize an empty list to store the layers of the neural network
        self.layers = []

        # Iterate through each layer in the JSON data
        # Note that the layer_name is queried but never used, idk but I'll keep it just in case something breaks
        for layer_name, parameters in data.items():
            # Extract the necessary layer parameters from the JSON data
            weights = np.array(parameters["weights"])
            biases = np.array(parameters["biases"])
            previousLayer_size = parameters["previousLayer_size"]
            layer_size = parameters["layer_size"]
            layer_type = parameters["layer_type"]
            activation_func_title = parameters["activation_func"]

            # Create a new instance of the Layer class with the appropriate parameters
            layer = Layer(previousLayer_size, layer_size, layer_type)

            # Load the weights and biases into the layer instance
            layer.load_weights_and_biases(weights, biases)

            # Set the activation function for the layer based on its title
            # Use the 'get_activation_function' method from the ActivationFunction class
            # Note that technically this is uneeded as the activation function is already set by default in the Layer class
            layer.set_activation_func(ActivationFunction.get_activation_function(activation_func_title))

            # Add the initialized layer to the neural network's list of layers
            self.add_layer(layer)
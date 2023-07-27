import numpy as np
from activation_functions import ActivationFunction

class Layer:
    # We're going to use leaky relu by default, but this can change!!!
    # Each "layer" consists of a set of nodes, and connections to the previous layer's nodes
    def __init__(self, previousLayer_size, layer_size, layer_type, activation_func=ActivationFunction.leaky_relu):
        # The layer size is the number of nodes in this layer
        # The previous layer size is the number of nodes in the previous layer
        self.previousLayer_size = previousLayer_size
        self.layer_size = layer_size

        self.layer_type = layer_type
        self.activation_func = activation_func

        # Initialize the layer's weights and biases randomly
        # Weights are a 2D array of size (output_size, layer_size) and biases are a 1D array of size (output_size)
        # Each array element in weights and element in bias correspond to each node
        # Weights are initialized randomly using a normal distribution with mean 0 and standard deviation 1
        # Biases are initialized to 0
        std = 0.01  # Your desired standard deviation
        self.weights = np.random.randn(self.layer_size, self.previousLayer_size) * std
        # self.weights = np.zeros((self.layer_size, self.previousLayer_size))
        self.biases = np.zeros(self.layer_size)
        if (self.layer_type == 'input'):
            self.weights = np.zeros((self.layer_size, self.previousLayer_size))

        # I don't know why the .derivative property doesnt wokr but i'll figure it out later
        self.activation_func.derivative = ActivationFunction.leaky_relu_derivative

        if (self.layer_type == 'output'):
            self.activation_func = ActivationFunction.tanh
            self.activation_func.derivative = ActivationFunction.tanh_derivative

        # Variable to store the weighted input and inputs for this layer
        self.weighted_input = None
        self.input_data = None

    # Load the weights and biases for this layer from something like a JSON file
    def load_weights_and_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases

    # Set the activation function for this layer if needed, like from a JSON file
    def set_activation_func(self, activation_func):
        self.activation_func = activation_func

    # Compute the output of this layer given the input data
    def compute_propogation(self, input_data):
        # Compute the net input for this layer
        # When we dot the weights matrix with the input data vector, we get a vector with a size that is the other matrix dimension
        # For example, if the weights matrix is 2x3 (2 high, 3 long) and the input data vector is 1x3 (1 high, 3 long)
        # Then the dot product of the matrix dotted with the vector (IN THAT ORDER) will be a 1x2 vector (1 high, 2 long)
        # Let's let the first element represent the first neuron, and so on
        weighted_input = np.dot(self.weights, input_data) + self.biases

        # Save the weighted input and inputs for this layer
        self.weighted_input = weighted_input
        self.input_data = input_data

        # Apply the activation function based on the layer type
        # Activation function normalizes the output of the layer
        if self.layer_type == 'input': # Input layer is just the input data
            output = input_data  # Weights/biases aren't applied since its just nodes, no connections to a nonexistent previous layer
        elif self.layer_type == 'output': # Note that output and hidden layers are computationally the same, but we differentiate them for clarity
            output = self.activation_func(weighted_input)  # Apply activation for output layer
        elif self.layer_type == 'hidden':
            output = self.activation_func(weighted_input)  # Apply activation for hidden layers
        else:
            raise ValueError("Invalid layer type.")

        return output
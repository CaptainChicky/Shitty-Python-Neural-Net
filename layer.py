import numpy as np
from activation_functions import ActivationFunction

# TODO:
# Add support for different activation functions that you can set customly
# Be able to set the .derivative attribute of the activation function in the activation function class (see that file)
# Implement softmax activation function for the output layer

class Layer:

    # We're going to use leaky relu by default, but this can change!!!
    # Each "layer" consists of a set of nodes, and connections to the previous layer's nodes
    def __init__(self, previousLayer_size, layer_size, layer_type, activation_func=ActivationFunction.leaky_relu):

        # The layer size is the number of nodes in this layer
        # The previous layer size is the number of nodes in the previous layer
        self.previousLayer_size = previousLayer_size
        self.layer_size = layer_size

        # The layer type is either input, hidden, or outpu
        self.layer_type = layer_type

        # The activation function is the function that is applied to the weighted input for each node, which is hardcoded to be leaky relu
        # The activation function.derivative is the derivative of the activation function, which is hardcoded to be leaky relu as well
        # Currently, I have to manually set the derivative, but in the future, this should be updated
        self.activation_func = activation_func
        self.activation_func.derivative = ActivationFunction.leaky_relu_derivative

        # We want to normalize the output layer's output to be between -1 and 1, so we use tanh for the output layer
        # Again, we're manually setting everything, but this should be updated in the future to allow for custom activation functions
        if (self.layer_type == 'output'):
            self.activation_func = ActivationFunction.tanh
            self.activation_func.derivative = ActivationFunction.tanh_derivative

        # Initialize the layer's weights
        # Weights are a 2D array of size (output_size, layer_size)
        # Each array element in weights correspond to each node's connections to the previous layer's nodes
        # Weights are initialized randomly using a normal distribution with mean 0 and standard deviation 0.01
        # Note that we're manually setting a standard deviation of 0.01, because if we use the default 1, we get exploding gradients
        std = 0.01  # Desired standard deviation
        self.weights = np.random.randn(self.layer_size, self.previousLayer_size) * std
        # Input layer has no effect, so we just set it to 0
        # We still want to initialize the weights for consistency, but they have no effect and are ignorable basicallly
        if (self.layer_type == 'input'):
            self.weights = np.zeros((self.layer_size, self.previousLayer_size))

        # Initialize the layer's biases
        # Biases are a 1D array of size (output_size)
        # Each array element in bias correspond to the bias of each node in the layer
        # Biases are initialized to 0
        self.biases = np.zeros(self.layer_size)

        # Variable to store the weighted input and inputs for this layer
        # This is used in backpropogation (see the training class)
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

        # Save the weighted input and inputs for this layer for backpropogation (see training class)
        self.weighted_input = weighted_input
        self.input_data = input_data

        # Apply the activation function based on the layer type
        if self.layer_type == 'input': # Input layer is just the input data
            output = input_data  # Weights/biases aren't applied since its just nodes, no connections to a nonexistent previous layer
            
        elif self.layer_type == 'output': # Note that output and hidden layers are computationally the same, but we differentiate them for clarity
            # Activation function normalizes the output of the layer to be between -1 and 1 if we are using tanh (which is hardcoded here for now)
            output = self.activation_func(weighted_input)  # Apply activation for output layer

        elif self.layer_type == 'hidden':
            output = self.activation_func(weighted_input)  # Apply activation for hidden layers

        else:
            raise ValueError("Invalid layer type.")

        return output
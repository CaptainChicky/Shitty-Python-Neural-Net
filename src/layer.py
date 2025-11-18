import numpy as np
import inspect
from activation_functions import ActivationFunction
from weight_initializers import WeightInitializer, BiasInitializer

# TODO:
# Add support for different activation functions that you can set customly
# Be able to set the .derivative attribute of the activation function in the activation function class (see that file)
# Implement softmax activation function for the output layer

class Layer:

    # We're going to use leaky relu by default, but this can change!!!
    # Each "layer" consists of a set of nodes, and connections to the previous layer's nodes
    def __init__(self, previousLayer_size, layer_size, layer_type,
                 activation_func=ActivationFunction.leaky_relu,
                 activation_params=None,
                 weight_init='he',
                 weight_init_params=None,
                 bias_init='zeros',
                 bias_init_params=None):

        # The layer size is the number of nodes in this layer
        # The previous layer size is the number of nodes in the previous layer
        self.previousLayer_size = previousLayer_size
        self.layer_size = layer_size

        # The layer type is either input, hidden, or output
        self.layer_type = layer_type

        # The activation function is the function that is applied to the weighted input for each node
        # We automatically set the derivative based on the activation function's title
        self.activation_func = activation_func

        # Store activation function parameters (e.g., alpha for leaky_relu, alpha for elu)
        #
        # NOTE: Even though activation functions have default parameters in activation_functions.py
        # (e.g., leaky_relu has alpha=0.01), we explicitly store those defaults here.
        # This is redundant, BUT it makes saved model JSON files self-documenting - you can see
        # exactly what alpha was used without having to look at the function definition.
        #
        # What happens:
        # - No custom params provided: Automatically extract defaults from function signature using inspect
        # - Custom params provided: Store custom ({'alpha': 0.05}), call leaky_relu(x, alpha=0.05)
        # - Non-parametric activation: Store empty dict ({}), call relu(x) with no params
        if activation_params is None:
            # Automatically extract default parameters from the activation function signature
            # This works for ANY parametric activation function without needing to hardcode each one!
            sig = inspect.signature(activation_func)
            self.activation_params = {
                name: param.default
                for name, param in sig.parameters.items()
                if param.default != inspect.Parameter.empty and name != 'x'
            }
        else:
            self.activation_params = activation_params  # User provided custom params

        # Automatically get the derivative function based on the activation function's title
        derivative_title = self.activation_func.title + "_derivative"
        self.activation_func.derivative = ActivationFunction.get_activation_function(derivative_title)

        # We want to normalize the output layer's output to be between -1 and 1, so we use tanh for the output layer
        if (self.layer_type == 'output'):
            self.activation_func = ActivationFunction.tanh
            self.activation_params = {}  # tanh has no parameters
            derivative_title = self.activation_func.title + "_derivative"
            self.activation_func.derivative = ActivationFunction.get_activation_function(derivative_title)

        # Initialize the layer's weights
        # Weights are a 2D array of size (layer_size, previousLayer_size)
        # Each row corresponds to one neuron's connections to all neurons in the previous layer
        #
        # Weight initialization uses dict notation for parameters (consistent with activation_params)
        # Examples: weight_init='normal', weight_init_params={'std': 0.02}
        #           weight_init='xavier', weight_init_params={}
        # See weight_initializers.py for details on each strategy

        weight_shape = (self.layer_size, self.previousLayer_size)

        # Get the weight initializer function by title
        weight_init_func = None
        for func_name in dir(WeightInitializer):
            func = getattr(WeightInitializer, func_name)
            if callable(func) and hasattr(func, 'title') and func.title == weight_init:
                weight_init_func = func
                break

        if weight_init_func is None:
            raise ValueError(f"Unknown weight initialization strategy: '{weight_init}'")

        # Extract default parameters from function signature if not provided
        if weight_init_params is None:
            sig = inspect.signature(weight_init_func)
            weight_init_params = {
                name: param.default
                for name, param in sig.parameters.items()
                if param.default != inspect.Parameter.empty and name not in ['shape', 'fan_in', 'fan_out']
            }

        # Special handling for initializers that need fan_in/fan_out
        if weight_init == 'xavier' or weight_init == 'uniform_xavier':
            self.weights = weight_init_func(weight_shape, previousLayer_size, layer_size, **weight_init_params)
        elif weight_init == 'he':
            self.weights = weight_init_func(weight_shape, previousLayer_size, **weight_init_params)
        else:
            # Normal, uniform, etc.
            self.weights = weight_init_func(weight_shape, **weight_init_params)

        # Input layer weights are always set to zero (they're not used)
        if self.layer_type == 'input':
            self.weights = np.zeros(weight_shape)

        # Initialize the layer's biases
        # Biases are a 1D array of size (layer_size)
        # Each element corresponds to the bias of one neuron in this layer
        #
        # Bias initialization uses dict notation for parameters (consistent with activation_params)
        # Examples: bias_init='zeros', bias_init_params={}
        #           bias_init='constant', bias_init_params={'value': 0.5}
        # See weight_initializers.py for details on each strategy

        bias_shape = self.layer_size

        # Get the bias initializer function by title
        bias_init_func = None
        for func_name in dir(BiasInitializer):
            func = getattr(BiasInitializer, func_name)
            if callable(func) and hasattr(func, 'title') and func.title == bias_init:
                bias_init_func = func
                break

        if bias_init_func is None:
            raise ValueError(f"Unknown bias initialization strategy: '{bias_init}'")

        # Extract default parameters from function signature if not provided
        if bias_init_params is None:
            sig = inspect.signature(bias_init_func)
            bias_init_params = {
                name: param.default
                for name, param in sig.parameters.items()
                if param.default != inspect.Parameter.empty and name != 'shape'
            }

        # Initialize biases with parameters
        self.biases = bias_init_func(bias_shape, **bias_init_params)

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
            # Pass activation parameters if they exist
            if self.activation_params:
                output = self.activation_func(weighted_input, **self.activation_params)
            else:
                output = self.activation_func(weighted_input)

        elif self.layer_type == 'hidden':
            # Pass activation parameters if they exist
            if self.activation_params:
                output = self.activation_func(weighted_input, **self.activation_params)
            else:
                output = self.activation_func(weighted_input)

        else:
            raise ValueError("Invalid layer type.")

        return output
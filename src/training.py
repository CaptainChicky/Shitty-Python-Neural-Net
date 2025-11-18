import numpy as np

class Training:
    
    # For the constructor, we're passing in normal things like the net itself, and the learning rate
    # However, we also have a clip value, which is used to prevent the gradient from exploding during backpropagation
    # This sets a hard cap on the gradient, so that it can't be too large (check the update parameters function in this class)
    #
    # cost_function options: 'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy'
    # See compatibility notes below for which cost function to use with which output activation
    # checkpoint_path: Optional path to save best model during training (None = no checkpointing)
    def __init__(self, neural_net, learning_rate, clip_value, cost_function='mse', checkpoint_path=None):
        self.neural_net = neural_net
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.cost_function = cost_function if cost_function is not None else 'mse'
        self.checkpoint_path = checkpoint_path
        self.best_cost = float('inf')  # Track best cost for checkpointing

        # Validate cost function and output activation pairing
        # In case someone retarded uses it i guess
        #==========================================================================
        output_layer = neural_net.layers[-1]
        output_activation = output_layer.activation_func.title

        # RULE 1: Binary Cross-Entropy REQUIRES Sigmoid activation
        if self.cost_function == 'binary_crossentropy':
            if output_activation != 'sigmoid':
                raise ValueError(
                    f"Binary cross-entropy loss REQUIRES sigmoid output activation.\n"
                    f"Current output activation: '{output_activation}'\n"
                )

        # RULE 2: Categorical Cross-Entropy REQUIRES Softmax activation
        if self.cost_function == 'categorical_crossentropy':
            if output_activation != 'softmax':
                raise ValueError(
                    f"Categorical cross-entropy loss REQUIRES softmax output activation.\n"
                    f"Current output activation: '{output_activation}'\n"
                )

        # RULE 3: Softmax REQUIRES Categorical Cross-Entropy
        if output_activation == 'softmax':
            if self.cost_function != 'categorical_crossentropy':
                raise ValueError(
                    f"Softmax output activation REQUIRES categorical cross-entropy loss.\n"
                    f"Current cost function: '{self.cost_function}'\n"
                )
        #==========================================================================

    # These are our cost functions and friends
    #
    # COST FUNCTION COMPATIBILITY:
    # - MSE (Mean Squared Error): Works with any output activation (tanh, sigmoid, softmax, etc.)
    # - MAE (Mean Absolute Error): Works with any output activation
    # - Binary Cross-Entropy: Best with sigmoid output (0-1 range), can work with tanh if data is [-1, 1]
    # - Categorical Cross-Entropy: Best with softmax output (probability distribution)

    # Node-wise cost functions (operate on individual nodes in the output layer)
    def node_cost(self, predicted_value, target_value):
        if self.cost_function == 'mse':
            # Mean Squared Error: 0.5 * (predicted - target)^2
            # The 0.5 makes the derivative cleaner
            return 0.5 * (predicted_value - target_value) ** 2

        elif self.cost_function == 'mae':
            # Mean Absolute Error: |predicted - target|
            return np.abs(predicted_value - target_value)

        elif self.cost_function == 'binary_crossentropy':
            # Binary Cross-Entropy: -[target * log(predicted) + (1-target) * log(1-predicted)]
            # Clip predictions to avoid log(0)
            eps = 1e-15
            predicted_clipped = np.clip(predicted_value, eps, 1 - eps)
            return -(target_value * np.log(predicted_clipped) + (1 - target_value) * np.log(1 - predicted_clipped))

        elif self.cost_function == 'categorical_crossentropy':
            # Categorical Cross-Entropy: -sum(target * log(predicted))
            # Clip predictions to avoid log(0)
            eps = 1e-15
            predicted_clipped = np.clip(predicted_value, eps, 1.0)
            return -(target_value * np.log(predicted_clipped))

        else:
            raise ValueError(f"Unknown cost function: '{self.cost_function}'")

    def node_cost_derivative(self, predicted_value, target_value):
        if self.cost_function == 'mse':
            # Derivative of MSE: predicted - target
            return predicted_value - target_value

        elif self.cost_function == 'mae':
            # Derivative of MAE: sign(predicted - target)
            # At exactly predicted == target, derivative is technically undefined, so we use 0
            diff = predicted_value - target_value
            return np.where(diff == 0, 0, np.sign(diff))

        elif self.cost_function == 'binary_crossentropy':
            # Derivative of Binary Cross-Entropy: (predicted - target) / [predicted * (1 - predicted)]
            # When combined with sigmoid activation, this simplifies to just (predicted - target)
            # For numerical stability, clip predictions
            eps = 1e-15
            predicted_clipped = np.clip(predicted_value, eps, 1 - eps)
            return (predicted_clipped - target_value) / (predicted_clipped * (1 - predicted_clipped))

        elif self.cost_function == 'categorical_crossentropy':
            # NOTE: This should NEVER be reached in correct usage!
            # Categorical cross-entropy should ONLY be used with softmax activation,
            # and softmax + categorical CE uses a hardcoded special case in
            # firstTwoDerivativesOfOutputLayer() that returns predicted - target directly.
            # If we reach here, it means categorical CE is being used with a non-softmax
            # activation (sigmoid, tanh, relu, etc.) which is INCORRECT.
            raise ValueError(
                "Categorical cross-entropy should ONLY be used with softmax output activation. "
                "The derivative is computed via a special case in backpropagation. "
                "If you're seeing this error, you likely used categorical_crossentropy with "
                "a non-softmax activation function, which produces incorrect gradients. "
                "Use 'binary_crossentropy' with sigmoid for binary classification, or "
                "'softmax' activation with 'categorical_crossentropy' for multi-class classification."
            )

        else:
            raise ValueError(f"Unknown cost function: '{self.cost_function}'")

    # We want to minimize the cost function, because the cost represents the error of the neural network
    # We only use this function for training transparency, so we can see how the cost changes over time
    # Otherwise, we don't really need this function
    def cost(self, predicted_values, target_values):
        # The cost function is the average of the node-wise cost functions for all nodes and all samples
        # For a single sample, predicted_values and target_values are 1D arrays
        # The node_cost method handles both element-wise (for arrays) and single values

        # Calculate the node-wise cost
        node_cost_values = self.node_cost(predicted_values, target_values)

        # Calculate the total cost by taking the mean of node-wise costs
        # This works whether we have a single sample (1D array) or multiple samples
        total_cost = np.mean(node_cost_values)

        return total_cost


    # Functions for backpropogation
    # Watch 3Blue1Brown and Sebastian Lague's video(s) on neural nets to get a good overview on this
    # We define these values:
    #   Weighted sum of inputs to a node: Z = W * A + B
    #   Input data: A = f(Z)

    # First two derivatives to use during backpropogation, which we set in this function for clarity
    # It is basically dCost/dPredictedValue * dPredictedValue/dZ
    # (where Z = weighted sum of inputs to the node without activation function applied)
    # This calculates the first two derivatives for the entire output layer
    # The predicted and target values are all 1D numpy arrays, so is the output
    def firstTwoDerivativesOfOutputLayer(self, predicted_values, target_values):
        # Values are queried from top down, so from the top node to the bottom node
        # Numpy can do the multiplication within the arrays for us lmao

        output_layer = self.neural_net.layers[-1]

        # SPECIAL CASE: Softmax + Categorical Cross-Entropy
        # When softmax activation is combined with categorical cross-entropy loss,
        # the full derivative (including the Jacobian-vector product) simplifies to: predicted - target
        # This is mathematically correct because:
        #   - CE derivative: -target / predicted
        #   - Softmax Jacobian-vector product: When multiplied with CE derivative, simplifies to: predicted - target
        # This is a well-known result in deep learning and avoids numerical instability
        # NOTE: This special case is REQUIRED because softmax has a Jacobian matrix derivative (not element-wise)
        if (output_layer.activation_func.title == 'softmax' and
            self.cost_function == 'categorical_crossentropy'):
            derivatives = predicted_values - target_values
            return derivatives

        # GENERAL CASE: Element-wise activations (sigmoid, tanh, relu, leaky_relu, elu, etc.)
        # For all other activation + cost function combinations, use standard chain rule
        # with element-wise multiplication (Hadamard product)
        # This includes sigmoid + binary cross-entropy, which naturally simplifies to predicted - target

        # The dCost/dPredictedValue term is simply the node cost derivative
        dCost_dPredictedValue = self.node_cost_derivative(predicted_values, target_values)

        # The dPredictedValue/dZ term is the derivative of the activation function with the weighted sum of that layer as input
        # We want to find the activation function of that layer, then find the derivative of that function
        # Then we plug in the weighted sum of that node into the derivative of the activation function for all output nodes
        if output_layer.activation_params:
            dPredictedValue_dZ = output_layer.activation_func.derivative(output_layer.weighted_input, **output_layer.activation_params)
        else:
            dPredictedValue_dZ = output_layer.activation_func.derivative(output_layer.weighted_input)

        # We want element-wise multiplication (Hadamard product) which is the point-wise multiplication of two arrays of the same shape
        # In numpy, we can achieve this by simply using the * operator between two arrays via numpy
        derivatives = dCost_dPredictedValue * dPredictedValue_dZ
        return derivatives
    

    # In the backward pass, the algorithm starts from the output layer and works backward through the network
    # It calculates the gradient of the cost function with respect to the weights and biases at each layer using the chain rule
    # Note that this function is only for a single sample
    # Returns both gradients and predicted_values to avoid redundant forward propagation
    def backpropagation(self, input_data, target_values):

        gradients = {}  # Dictionary to store gradients for each layer's weights and biases

        #############################
        # Forward pass              #
        #############################
        # Perform forward propogation to get the predicted values from the neural network
        predicted_values = self.neural_net.forward_propagation(input_data)


        #############################
        # Backward pass             #
        #############################

        # Calculate the gradients for the !!OUTPUT LAYER!! ONLY #

        # Weights
        # For each weight in the output layer, we want to multiply the first two derivatives of the output layer's corresponding node
        First2Dervs = self.firstTwoDerivativesOfOutputLayer(predicted_values, target_values)
        
        # Preallocate the output weight gradients with zeros
        OutputWeightGradients = np.zeros_like(self.neural_net.layers[-1].weights)

        # We want to calculate dZ/dWeight for each weight in the output layer
        # dZ/dWeight = input data that corresponds to each weight
        # That input data would just be the data each node in the this layer recieves
        # 
        # We're going to use the outer product to calculate the output weight gradients
        # The result will be a 2D array with the same shape as self.neural_net.layers[-1].weights
        # Explanation: 
        #   Each "first2derivs" are multiplied by each respective activation value in the previous layer
        #   Each of the aforementioned list is an element of our weight gradient list
        #   The outer product will then generate the matrix of all possible combinations of the two lists, which is the same shape as the weights
        OutputWeightGradients = np.outer(First2Dervs, self.neural_net.layers[-1].input_data)

        # Append the output weight gradients to the dictionary
        gradients[f"weights_{len(self.neural_net.layers) - 1}"] = OutputWeightGradients

        # Biases
        # Biases gradients are simply the first two derivatives lmao since the final deriv is 1
        gradients[f"biases_{len(self.neural_net.layers) - 1}"] = First2Dervs

        
        # Calculate the gradients for the !!HIDDEN LAYERS!! #

        # We want to iterate through the hidden layers in reverse order
        # Note that we exclude the input layer because the input layer doesn't do any calculations and just passes on its input

        # Next derivatives store the running derivative multiplication count as we go down layers
        next_derivatives = First2Dervs

        # Iterate through the hidden layers in reverse order, while multiplying the derivatives of the next layer with the current layer's derivatives
        for layer_idx in range(len(self.neural_net.layers) - 2, 0, -1): # Exclude input and output layers 

            # Set some useful variables for the current and next layer to avoid calling them manually and for clarity
            current_layer = self.neural_net.layers[layer_idx]
            next_layer = self.neural_net.layers[layer_idx + 1]
            
            # Refresher: 
            #   Let z = weight * input + bias
            #   Let a be the input ActivationFunction(weighted_input)
            # We're trying to find the derivative of the dZ/dA of this layer, which is the next layer's weights
            dZ_dActivation = next_layer.weights

            # Now we're taking the activation of this layer and taking its derivative with respect to the the same layer's weighted inputs
            # This is the derivative of the activation function with this layer's weighted input as a argument
            if current_layer.activation_params:
                dActivation_dZprev = current_layer.activation_func.derivative(current_layer.weighted_input, **current_layer.activation_params)
            else:
                dActivation_dZprev = current_layer.activation_func.derivative(current_layer.weighted_input)

            # We multiply this one by one to the next derivatives, first the dZ/dA then the dA/dZ_next
            # The dot product of the next derivatives and dZ_dActivation will give a vector with the size of the current layer's nodes
            # Then we perform the Hadamard product with the dActivation_dZprev to get the final vector of the current layer's nodes
            next_derivatives = np.dot(next_derivatives, dZ_dActivation) * dActivation_dZprev

            # Calculate the weight gradients for the current layer using the outer product (basically same as the output layer)
            weight_gradients = np.outer(next_derivatives, current_layer.input_data)

            # Append the weight gradients and bias gradients to the dictionary
            gradients[f"weights_{layer_idx}"] = weight_gradients

            # Biases gradients are simply the next derivatives because the final derivative dZ/dB is 1
            gradients[f"biases_{layer_idx}"] = next_derivatives

        
        # Input layer should not change (it does not affect the neural net), so set all gradients to zero
        gradients[f"weights_0"] = np.zeros_like(self.neural_net.layers[0].weights)
        gradients[f"biases_0"] = np.zeros_like(self.neural_net.layers[0].biases)

        return gradients, predicted_values


    # Update the weights and biases of the neural network using the gradients obtained from backpropagation
    def update_parameters(self, gradients, clip_value):

        # Iterate through each layer in the neural network
        for i, layer in enumerate(self.neural_net.layers):

            # Get the gradients for the current layer
            weight_gradients = gradients[f"weights_{i}"]
            bias_gradients = gradients[f"biases_{i}"]

            # Clip gradients to prevent exploding gradients, so at most the gradients will be +/- the clip value
            weight_gradients = np.clip(weight_gradients, -clip_value, clip_value)
            bias_gradients = np.clip(bias_gradients, -clip_value, clip_value)

            # Then apply the gradients to the weights and biases of the current layer with respect to the learning rate
            layer.weights -= self.learning_rate * weight_gradients
            layer.biases -= self.learning_rate * bias_gradients
            

    # Train the neural network using the given input and target data for the given number of epochs
    # Remember that a single epoch is a single iteration of the entire training set (or a subset if samples_per_epoch is specified)
    # For each epoch, the neural net takes a small step towards a local minimum via gradient descent
    def train(self, input_data, target_data, epochs, samples_per_epoch=None):
        """
        Train the neural network.

        Args:
            input_data: Training input samples (numpy array)
            target_data: Training target values (numpy array)
            epochs: Number of training epochs
            samples_per_epoch: Number of samples to use per epoch (default None = use all data)
                              If specified, randomly samples this many data points each epoch.
                              This helps prevent overfitting and adds regularization.
        """

        for epoch in range(epochs):
            total_cost = 0.0  # Variable to store the total cost for the current epoch

            # Determine which samples to use for this epoch
            if samples_per_epoch is not None and samples_per_epoch < len(input_data):
                # Randomly sample a subset of indices for this epoch
                # Use permutation instead of choice for better performance (no duplicate checking needed)
                sample_indices = np.random.permutation(len(input_data))[:samples_per_epoch]
            else:
                # Use all data (original behavior)
                sample_indices = range(len(input_data))

            # Iterate over each selected data point for this epoch
            for i in sample_indices:

                # Get the current input and target data sample
                input_sample = input_data[i]
                target_sample = target_data[i]

                # Compute gradients using backpropagation (which also returns predicted values)
                gradients, predicted_values = self.backpropagation(input_sample, target_sample)

                # Calculate the cost for the current sample and add it to the total cost for this epoch
                sample_cost = self.cost(predicted_values, target_sample)
                total_cost += sample_cost

                # Update the parameters (weights and biases) using the computed gradients
                self.update_parameters(gradients, self.clip_value)

            # Calculate the average cost for this epoch and print it
            num_samples_used = len(sample_indices)
            avg_cost = total_cost / num_samples_used

            # Checkpoint: Save model if this is the best cost so far
            if self.checkpoint_path is not None and avg_cost < self.best_cost:
                self.best_cost = avg_cost
                self.neural_net.save(self.checkpoint_path)
                checkpoint_msg = f" - NEW BEST! Saved to {self.checkpoint_path}"
            else:
                checkpoint_msg = ""

            # Show how many samples were used if using subset training
            if samples_per_epoch is not None and samples_per_epoch < len(input_data):
                print(f"Epoch {epoch + 1}/{epochs}, Average Cost: {avg_cost} (trained on {num_samples_used}/{len(input_data)} samples){checkpoint_msg}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Average Cost: {avg_cost}{checkpoint_msg}")
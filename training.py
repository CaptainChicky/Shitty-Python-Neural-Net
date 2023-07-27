import numpy as np
from layer import Layer
from neural_network import NeuralNet

# TODO:
# Change the error from MSE to cross-entropy, or at least give the ability to choose
# Activation function derivative shenanigans (see the activation functions class)
# Training function is not that efficient, because we're doing forward propagation twice, so optimize this (see that function)
# Cost function is inefficiently built, because it's built to average out the entire data set, but we're only using it for a single sample

class Training:
    
    # For the constructor, we're passing in normal things like the net itself, and the learning rate
    # However, we also have a clip value, which is used to prevent the gradient from exploding during backpropagation
    # This sets a hard cap on the gradient, so that it can't be too large (check the update parameters function in this class)
    def __init__(self, neural_net, learning_rate, clip_value):
        self.neural_net = neural_net
        self.learning_rate = learning_rate
        self.clip_value = clip_value


    # These are our cost functions and friends

    # This function only calculates the MSE cost for a single node (in the output layer, ideally)
    # Maybe use this for the cross-entropy cost function? See the TODO above
    @staticmethod
    def node_cost(predicted_value, target_value):
        # The node-wise (each node) cost function computes the squared error between the predicted and target values
        # We're multiplying by 0.5 to make the derivative of the cost function simpler
        return 0.5 * (predicted_value - target_value) ** 2

    @staticmethod
    def node_cost_derivative(predicted_value, target_value):
        # The derivative of the node-wise cost function is simply the difference between the predicted and target values
        # We'll be accepting numpy arrays as inputs, so we'll be returning a numpy array as output
        return predicted_value - target_value 

    # We want to minimize the cost function, because the cost represents the error of the neural network
    # We only use this function for training transparency, so we can see how the cost changes over time
    # Otherwise, we don't really need this function
    @staticmethod
    def cost(predicted_values, target_values):
        # The cost function is the average of the node-wise cost functions for all nodes and all samples in a training set
        # predicted_values and target_values are both 2D NUMPY ARRAYS
        # Each inner array represents the predicted values or target values for a given sample

        node_costs = []  # List to store the node-wise costs for each node in each sample

        # Iterate through each sample in the training set
        for i in range(len(predicted_values)):
            # Calculate the node-wise cost for the current sample
            node_cost = Training.node_cost(predicted_values[i], target_values[i])
            
            # Add the node-wise cost for the current sample to the list of node-wise costs
            node_costs.append(node_cost) 

        # Calculate the total cost by taking the mean of node-wise costs over all nodes and all samples in a given training set
        # Remember that a single epoch is a single pass through the entire training set, so we're calculating the cost for a single epoch
        total_cost = np.mean(node_costs)

        # Note that there may be a problem here because in the actual training function where we use this, we're only using it for a single sample
        # This means we may not need to use the mean then
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

        # The dCost/dPredictedValue term is simply the node cost derivative
        dCost_dPredictedValue = self.node_cost_derivative(predicted_values, target_values)

        # The dPredictedValue/dZ term is the derivative of the activation function with the weighted sum of that layer as input
        # We want to find the activation function of that layer, then find the derivative of that function
        # Then we plug in the weighted sum of that node into the derivative of the activation function for all output nodes
        dPredictedValue_dZ = self.neural_net.layers[-1].activation_func.derivative(self.neural_net.layers[-1].weighted_input)

        # We want element-wise multiplication (Hadamard product) which is the point-wise multiplication of two arrays of the same shape
        # In numpy, we can achieve this by simply using the * operator between two arrays via numpy
        derivatives = dCost_dPredictedValue * dPredictedValue_dZ
        return derivatives
    

    # In the backward pass, the algorithm starts from the output layer and works backward through the network
    # It calculates the gradient of the cost function with respect to the weights and biases at each layer using the chain rule
    # Note that this function is only for a single sample
    def backpropagation(self, input_data, target_values):
        
        gradients = {}  # Dictionary to store gradients for each layer's weights and biases

        #############################
        # Forward pass              #
        #############################
        # Perform we perform a forward propogation to get the predicted values from the neural network
        # This isn't needed if you want to use the predicted values as the input, but I'm doing this just because lmao idk
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

        return gradients


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
    # Remember that a single epoch is a single iteration of the entire training set
    # For each epoch, the neural net takes a small step towards a local minimum via gradient descent
    def train(self, input_data, target_data, epochs):

        for epoch in range(epochs):
            total_cost = 0.0  # Variable to store the total cost for the current epoch
            
            # Iterate over each data point in the training set (looping through the entire training set)
            for i in range(len(input_data)):

                # Get the current input and target data sample
                input_sample = input_data[i]
                target_sample = target_data[i]

                # Forward pass to get the predicted values from the neural network for the current input sample
                # This may not be the most efficient way to do it, because we're doing forward propagation twice
                # Once here, so we have an input into the cost, and once again in the backpropagation function
                predicted_values = self.neural_net.forward_propagation(input_sample)

                # Calculate the cost for the current sample and add it to the total cost for this epoch
                # Optimize this (see TODO list)
                sample_cost = self.cost(predicted_values, target_sample)
                total_cost += sample_cost

                # Compute gradients using backpropagation
                gradients = self.backpropagation(input_sample, target_sample)

                # Update the parameters (weights and biases) using the computed gradients
                self.update_parameters(gradients, self.clip_value)

            # Calculate the average cost for this epoch and print it
            avg_cost = total_cost / len(input_data)
            print(f"Epoch {epoch + 1}/{epochs}, Average Cost: {avg_cost}")
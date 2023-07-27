import numpy as np

# TODO:
# Implement the ".derivative" method for each activation function
# Allow the custom setting of alpha in leaky relu, as currently it is hardcoded to 0.01
# Implement softmax activation function for the output layer

class ActivationFunction:
    # Custom decorator to add a "title" attribute to the function
    @staticmethod
    def with_title(title):
        # The `with_title` method takes a `title` argument and returns a decorator function

        def decorator(func):
            # The `decorator` function is the actual decorator that wraps the original `func`
            # When a function is decorated with `with_title`, it adds the `title` attribute to the function
            func.title = title
            # Return the original function (`func`) after adding the `title` attribute to it
            return func

        # Return the decorator function, so it can be used to modify other functions
        return decorator


    # Possible activation functions
    # All functions accept arrays because they are implemented using numpy
    @staticmethod
    @with_title("sigmoid")
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    @with_title("sigmoid_derivative")
    def sigmoid_derivative(x):
        # Compute the sigmoid using the previously defined sigmoid function
        sigmoid_x = ActivationFunction.sigmoid(x)
        
        # Calculate the derivative directly using the sigmoid value
        return sigmoid_x * (1 - sigmoid_x)


    @staticmethod
    @with_title("sign")
    def sign(x):
        return np.sign(x)
    
    @staticmethod
    @with_title("sign_derivative")
    def sign_derivative(x):
        return np.zeros_like(x)
    

    @staticmethod
    @with_title("step")
    def step(x):
        return np.where(x > 0, 1, 0.5)
        
    @staticmethod
    @with_title("step_derivative")
    def step_derivative(x):
        return np.zeros_like(x)
    

    @staticmethod
    @with_title("relu")
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    @with_title("relu_derivative")
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Note that leaky relu avoids the problem of dying neurons of relu, where the neuron stops learning because the gradient is 0
    # You can set alpha to a small value like 0.01, as it is currently harcoded to
    @staticmethod
    @with_title("leaky_relu")
    def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)
    
    @staticmethod
    @with_title("leaky_relu_derivative")
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


    @staticmethod
    @with_title("tanh")
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    @with_title("tanh_derivative")
    def tanh_derivative(x):
        return 1 / (np.cosh(x) ** 2)
    
    

    # Function to get the activation function based on its title
    @staticmethod
    def get_activation_function(title):
        # Loop through all the attributes (functions and other properties) of the ActivationFunction class
        for func in dir(ActivationFunction):
            # Get the attribute (function or property) using its name 'func' from the ActivationFunction class
            attr = getattr(ActivationFunction, func)    

            # Check if the attribute is a callable (function) and has the 'title' attribute
            # We only want to consider functions that are decorated with the 'with_title' decorator
            if callable(attr) and hasattr(attr, "title") and attr.title == title:
                # If the function has the correct 'title', return the function
                return attr 

        # If no matching activation function is found, raise a ValueError
        raise ValueError(f"Activation function with title '{title}' not found.")
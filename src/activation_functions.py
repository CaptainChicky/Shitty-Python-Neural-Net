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
        """
        Sigmoid activation function.
        Maps input to range (0, 1) with S-shaped curve.

        Args:
            x: Input array

        Returns:
            Output array with values in range (0, 1)
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    @with_title("sigmoid_derivative")
    def sigmoid_derivative(x):
        """
        Derivative of sigmoid activation function.

        Args:
            x: Input array

        Returns:
            Derivative values
        """
        # Compute the sigmoid using the previously defined sigmoid function
        sigmoid_x = ActivationFunction.sigmoid(x)

        # Calculate the derivative directly using the sigmoid value
        return sigmoid_x * (1 - sigmoid_x)


    @staticmethod
    @with_title("sign")
    def sign(x):
        """
        Sign activation function.
        Returns -1 for negative, 0 for zero, 1 for positive.

        Args:
            x: Input array

        Returns:
            Array of signs (-1, 0, or 1)
        """
        return np.sign(x)

    @staticmethod
    @with_title("sign_derivative")
    def sign_derivative(x):
        """
        Derivative of sign activation function.
        Always returns zero (sign is not differentiable).

        Args:
            x: Input array

        Returns:
            Array of zeros
        """
        return np.zeros_like(x)
    

    @staticmethod
    @with_title("step")
    def step(x):
        """
        Step activation function.
        Returns 1 for positive values, 0.5 for zero or negative.

        Args:
            x: Input array

        Returns:
            Array of step values (0.5 or 1)
        """
        return np.where(x > 0, 1, 0.5)

    @staticmethod
    @with_title("step_derivative")
    def step_derivative(x):
        """
        Derivative of step activation function.
        Always returns zero (step is not differentiable).

        Args:
            x: Input array

        Returns:
            Array of zeros
        """
        return np.zeros_like(x)
    

    @staticmethod
    @with_title("relu")
    def relu(x):
        """
        ReLU (Rectified Linear Unit) activation function.
        Returns max(0, x) - zeros out negative values.

        Args:
            x: Input array

        Returns:
            Array with negative values set to 0
        """
        return np.maximum(0, x)

    @staticmethod
    @with_title("relu_derivative")
    def relu_derivative(x):
        """
        Derivative of ReLU activation function.
        Returns 1 for positive values, 0 otherwise.

        Args:
            x: Input array

        Returns:
            Gradient array (0 or 1)
        """
        return np.where(x > 0, 1, 0)

    # Note that leaky relu avoids the problem of dying neurons of relu, where the neuron stops learning because the gradient is 0
    # You can set alpha to a small value like 0.01, as it is currently harcoded to
    @staticmethod
    @with_title("leaky_relu")
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation function.
        Like ReLU but allows small negative values (alpha * x for x < 0).
        Avoids dying neuron problem of standard ReLU.

        Args:
            x: Input array
            alpha: Slope for negative values (default 0.01)

        Returns:
            Array with leaky ReLU applied
        """
        return np.maximum(alpha * x, x)

    @staticmethod
    @with_title("leaky_relu_derivative")
    def leaky_relu_derivative(x, alpha=0.01):
        """
        Derivative of Leaky ReLU activation function.
        Returns 1 for positive values, alpha for negative.

        Args:
            x: Input array
            alpha: Slope for negative values (default 0.01)

        Returns:
            Gradient array (alpha or 1)
        """
        return np.where(x > 0, 1, alpha)

    # elu is smoother than leaky relu and can produce negative outputs, pushes mean activations closer to zero and can speed up learning
    # When x > 0: f(x) = x
    # When x <= 0: f(x) = alpha * (e^x - 1)
    @staticmethod
    @with_title("elu")
    def elu(x, alpha=1.0):
        """
        ELU (Exponential Linear Unit) activation function.
        Smoother than Leaky ReLU with exponential curve for negative values.
        Helps push mean activations closer to zero.

        Args:
            x: Input array
            alpha: Controls saturation for negative values (default 1.0)

        Returns:
            Array with ELU applied
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    @with_title("elu_derivative")
    def elu_derivative(x, alpha=1.0):
        """
        Derivative of ELU activation function.

        Args:
            x: Input array
            alpha: Controls saturation for negative values (default 1.0)

        Returns:
            Gradient array
        """
        # When x > 0: f'(x) = 1
        # When x <= 0: f'(x) = alpha * e^x = f(x) + alpha
        return np.where(x > 0, 1, alpha * np.exp(x))

    @staticmethod
    @with_title("tanh")
    def tanh(x):
        """
        Tanh (hyperbolic tangent) activation function.
        Maps input to range (-1, 1) with S-shaped curve.
        Zero-centered alternative to sigmoid.

        Args:
            x: Input array

        Returns:
            Output array with values in range (-1, 1)
        """
        return np.tanh(x)

    @staticmethod
    @with_title("tanh_derivative")
    def tanh_derivative(x):
        """
        Derivative of tanh activation function.

        Args:
            x: Input array

        Returns:
            Gradient array
        """
        return 1 / (np.cosh(x) ** 2)

    # Swish activation (also called silu aka Sigmoid Linear Unit) with configurable alpha
    # This is a smooth, non-monotonic activation function that can outperform relu
    # When alpha=1: f(x) = x * sigmoid(x) (standard Swish)
    # Higher alpha makes it more like relu, lower alpha makes it more linear
    @staticmethod
    @with_title("swish")
    def swish(x, alpha=1.0):
        """
        Swish/SiLU (Sigmoid Linear Unit) activation function.
        Smooth, non-monotonic activation that can outperform ReLU.
        Used in modern architectures like EfficientNet.

        Args:
            x: Input array
            alpha: Scaling factor for sigmoid (default 1.0)

        Returns:
            Array with Swish applied
        """
        return x * (1 / (1 + np.exp(-alpha * x)))

    @staticmethod
    @with_title("swish_derivative")
    def swish_derivative(x, alpha=1.0):
        """
        Derivative of Swish activation function.

        Args:
            x: Input array
            alpha: Scaling factor for sigmoid (default 1.0)

        Returns:
            Gradient array
        """
        # Derivative: f'(x) = f(x) + sigmoid(alpha*x) * (1 - f(x))
        # Or equivalently: f'(x) = alpha*f(x) + sigmoid(alpha*x)*(1 - alpha*f(x))
        sigmoid_val = 1 / (1 + np.exp(-alpha * x))
        swish_val = x * sigmoid_val
        return swish_val + sigmoid_val * (1 - swish_val)

    # SELU (Scaled Exponential Linear Unit) is a REAL 2-parameter activation function!
    # Used in Self-Normalizing Neural Networks (Klambauer et al., 2017)
    # These specific alpha and lambda values ensure self-normalizing properties
    # When x > 0: f(x) = lambda * x
    # When x <= 0: f(x) = lambda * alpha * (e^x - 1)
    @staticmethod
    @with_title("selu")
    def selu(x, alpha=1.67326324, scale=1.05070098):
        """
        SELU (Scaled Exponential Linear Unit) activation function.
        Real 2-parameter activation from Self-Normalizing Neural Networks paper.
        These specific parameter values ensure self-normalizing properties.

        Args:
            x: Input array
            alpha: Controls saturation for negative values (default 1.67326324)
            scale: Scaling factor (lambda) for entire function (default 1.05070098)

        Returns:
            Array with SELU applied
        """
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    @with_title("selu_derivative")
    def selu_derivative(x, alpha=1.67326324, scale=1.05070098):
        """
        Derivative of SELU activation function.

        Args:
            x: Input array
            alpha: Controls saturation for negative values (default 1.67326324)
            scale: Scaling factor (lambda) for entire function (default 1.05070098)

        Returns:
            Gradient array
        """
        # When x > 0: f'(x) = scale
        # When x <= 0: f'(x) = scale * alpha * e^x
        return scale * np.where(x > 0, 1, alpha * np.exp(x))

    # GELU (Gaussian Error Linear Unit) is used in BERT, GPT, and many transformer models
    # f(x) = x * Φ(x) where Φ(x) is the CDF of the standard normal distribution
    # We use the tanh approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    @staticmethod
    @with_title("gelu")
    def gelu(x):
        """
        GELU (Gaussian Error Linear Unit) activation function.
        Used in transformers like BERT and GPT.
        Smooth approximation to ReLU with probabilistic interpretation.

        Args:
            x: Input array

        Returns:
            Array with GELU applied
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    @with_title("gelu_derivative")
    def gelu_derivative(x):
        """
        Derivative of GELU activation function (tanh approximation).

        Args:
            x: Input array

        Returns:
            Gradient array
        """
        # Compute intermediate values
        sqrt_2_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_pi * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        sech2_inner = 1 - tanh_inner**2  # sech^2(x) = 1 - tanh^2(x)

        # Derivative of inner term
        inner_derivative = sqrt_2_pi * (1 + 3 * 0.044715 * x**2)

        # Full derivative
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2_inner * inner_derivative

    # Mish activation gives smooth, self-regularized non-monotonic activation
    # f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    # Can outperform ReLU and Swish in some deep learning tasks
    @staticmethod
    @with_title("mish")
    def mish(x):
        """
        Mish activation function.
        Smooth, self-regularized non-monotonic activation.
        Can outperform ReLU and Swish in some tasks.

        Args:
            x: Input array

        Returns:
            Array with Mish applied
        """
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    @with_title("mish_derivative")
    def mish_derivative(x):
        """
        Derivative of Mish activation function.

        Args:
            x: Input array

        Returns:
            Gradient array
        """
        # softplus(x) = ln(1 + e^x)
        # softplus'(x) = e^x / (1 + e^x) = sigmoid(x)
        # mish(x) = x * tanh(softplus(x))
        # mish'(x) = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)

        exp_x = np.exp(x)
        softplus = np.log(1 + exp_x)
        tanh_softplus = np.tanh(softplus)
        sigmoid_x = exp_x / (1 + exp_x)
        sech2_softplus = 1 - tanh_softplus**2

        return tanh_softplus + x * sech2_softplus * sigmoid_x

    # Parametric activation with TWO parameters (alpha and beta)
    # This is just for demonstration, no real things irl use this shit lol
    # f(x) = alpha * x  when x > 0
    # f(x) = beta * (e^x - 1)  when x <= 0
    # This combines features of both Leaky ReLU (alpha) and ELU (beta)
    @staticmethod
    @with_title("parametric_elu")
    def parametric_elu(x, alpha=1.0, beta=1.0):
        """
        Parametric ELU activation with two parameters.
        Combines features of Leaky ReLU (alpha) and ELU (beta).
        Demonstrates dict notation for multi-parameter activations.

        Args:
            x: Input array
            alpha: Slope for positive values (default 1.0)
            beta: Controls saturation for negative values (default 1.0)

        Returns:
            Array with parametric ELU applied
        """
        return np.where(x > 0, alpha * x, beta * (np.exp(x) - 1))

    @staticmethod
    @with_title("parametric_elu_derivative")
    def parametric_elu_derivative(x, alpha=1.0, beta=1.0):
        """
        Derivative of Parametric ELU activation function.

        Args:
            x: Input array
            alpha: Slope for positive values (default 1.0)
            beta: Controls saturation for negative values (default 1.0)

        Returns:
            Gradient array
        """
        # When x > 0: f'(x) = alpha
        # When x <= 0: f'(x) = beta * e^x
        return np.where(x > 0, alpha, beta * np.exp(x))
    
    
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
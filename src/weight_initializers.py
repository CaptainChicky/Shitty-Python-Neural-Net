import numpy as np

# Weight and bias initialization strategies for neural network layers
# These functions provide different ways to initialize weights and biases to help with training

class WeightInitializer:
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
    """
    Collection of weight initialization strategies.
    Each method returns a numpy array with the specified shape and initialization.
    """

    @staticmethod
    @with_title("normal")
    def normal(shape, std=0.01):
        """
        Normal (Gaussian) distribution initialization.

        Args:
            shape: Tuple specifying the shape of the weight matrix
            std: Standard deviation of the normal distribution (default 0.01)

        Returns:
            Numpy array with values sampled from N(0, std^2)
        """
        return np.random.randn(*shape) * std

    @staticmethod
    @with_title("xavier")
    def xavier(shape, fan_in, fan_out):
        """
        Xavier/Glorot initialization - good for sigmoid/tanh activations.
        Keeps the variance of activations and gradients roughly the same across layers.

        Formula: std = sqrt(2 / (fan_in + fan_out))

        Args:
            shape: Tuple specifying the shape of the weight matrix
            fan_in: Number of input units (previousLayer_size)
            fan_out: Number of output units (layer_size)

        Returns:
            Numpy array initialized with Xavier strategy
        """
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std

    @staticmethod
    @with_title("he")
    def he(shape, fan_in):
        """
        He initialization - specifically designed for ReLU/Leaky ReLU activations.
        Accounts for the fact that ReLU kills half the neurons (negative values become 0).

        Formula: std = sqrt(2 / fan_in)

        Note: This is the BEST choice for Leaky ReLU with small alpha (e.g., 0.01).
        The theoretical optimal for Leaky ReLU would be std = sqrt(2 / ((1 + alpha²) * fan_in)),
        but with alpha=0.01: (1 + 0.01²) = 1.0001 ≈ 1.0, making the difference negligible.
        Standard He init is the industry default for all ReLU-like activations.

        Args:
            shape: Tuple specifying the shape of the weight matrix
            fan_in: Number of input units (previousLayer_size)

        Returns:
            Numpy array initialized with He strategy
        """
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std
    
    @staticmethod
    @with_title("lecun_normal")
    def lecun_normal(shape, fan_in):
        """
        LeCun Normal initialization - designed for SELU and tanh activations.
        Keeps the variance of activations stable across layers.

        Formula: std = sqrt(1 / fan_in)

        This is REQUIRED for SELU to maintain its self-normalizing properties.
        Also works well with tanh activation.

        Args:
            shape: Tuple specifying the shape of the weight matrix
            fan_in: Number of input units (previousLayer_size)

        Returns:
            Numpy array initialized with LeCun Normal strategy
        """
        std = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * std

    @staticmethod
    @with_title("lecun_uniform")
    def lecun_uniform(shape, fan_in):
        """
        LeCun Uniform initialization - uniform distribution variant of LeCun init.
        Alternative to LeCun Normal, also designed for SELU and tanh.

        Formula: limit = sqrt(3 / fan_in)

        Args:
            shape: Tuple specifying the shape of the weight matrix
            fan_in: Number of input units (previousLayer_size)

        Returns:
            Numpy array with uniform LeCun initialization
        """
        limit = np.sqrt(3.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    @with_title("uniform")
    def uniform(shape, limit):
        """
        Uniform distribution initialization.

        Args:
            shape: Tuple specifying the shape of the weight matrix
            limit: Values will be sampled from [-limit, limit]

        Returns:
            Numpy array with values uniformly distributed in [-limit, limit]
        """
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    @with_title("uniform_xavier")
    def uniform_xavier(shape, fan_in, fan_out):
        """
        Xavier/Glorot uniform initialization.
        Alternative to Xavier normal distribution.

        Formula: limit = sqrt(6 / (fan_in + fan_out))

        Args:
            shape: Tuple specifying the shape of the weight matrix
            fan_in: Number of input units
            fan_out: Number of output units

        Returns:
            Numpy array with uniform Xavier initialization
        """
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)


class BiasInitializer:
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

    """
    Collection of bias initialization strategies.
    Biases are typically initialized much more simply than weights.
    """

    @staticmethod
    @with_title("zeros")
    def zeros(shape):
        """
        Initialize all biases to zero (most common approach).

        Args:
            shape: Size of the bias vector

        Returns:
            Numpy array of zeros
        """
        return np.zeros(shape)

    @staticmethod
    @with_title("ones")
    def ones(shape):
        """
        Initialize all biases to one (rarely used).

        Args:
            shape: Size of the bias vector

        Returns:
            Numpy array of ones
        """
        return np.ones(shape)

    @staticmethod
    @with_title("constant")
    def constant(shape, value):
        """
        Initialize all biases to a specific constant value.
        Sometimes used for LSTM forget gates (often set to 1.0).

        Args:
            shape: Size of the bias vector
            value: Constant value to fill the array with

        Returns:
            Numpy array filled with the constant value
        """
        return np.full(shape, value)

    @staticmethod
    @with_title("normal")
    def normal(shape, std=0.01):
        """
        Initialize biases with small random values from normal distribution.
        Rarely used - zeros is almost always better.

        Args:
            shape: Size of the bias vector
            std: Standard deviation of the normal distribution

        Returns:
            Numpy array with values sampled from N(0, std^2)
        """
        return np.random.randn(shape) * std


import numpy as np
import struct
import json
import os
from array import array

class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        # Read labels
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())        
        
        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())        
        
        # Reshape images
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images.append(img)  # Keep as 1D array (784 pixels)
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, 
            self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, 
            self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


def label_to_onehot(label):
    """Convert digit 0-9 to one-hot encoding [0,0,1,0,0,0,0,0,0,0] for digit 2"""
    onehot = [0.0] * 10
    onehot[label] = 1.0
    return onehot


def prepare_mnist_for_neural_net(x_data, y_data, num_samples=None):
    """
    Convert MNIST to neural network format:
    - Flatten images to 784 pixels
    - Normalize pixels from [0, 255] to [0, 1]
    - Convert labels to one-hot encoding
    """
    if num_samples is None:
        num_samples = len(x_data)
    
    input_data = []
    output_data = []
    
    for i in range(num_samples):
        # Normalize pixels to [0, 1]
        normalized_pixels = [float(pixel) / 255.0 for pixel in x_data[i]]
        input_data.append(normalized_pixels)
        
        # Convert label to one-hot
        onehot = label_to_onehot(int(y_data[i]))
        output_data.append(onehot)
    
    return input_data, output_data


if __name__ == "__main__":
    print("=" * 70)
    print("Loading MNIST Dataset...")
    print("=" * 70)
    
    # Get paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up to project root
    mnist_dir = os.path.join(project_root, 'data', 'mnist')
    
    # Set file paths using relative paths
    training_images_filepath = os.path.join(mnist_dir, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(mnist_dir, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(mnist_dir, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')
    
    print(f"\n📁 Looking for MNIST files in: {mnist_dir}")
    
    # Load MNIST
    print("\nLoading MNIST binary files...")
    loader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    
    print(f"✅ Loaded {len(x_train)} training images")
    print(f"✅ Loaded {len(x_test)} test images")
    print(f"   Image size: 28x28 = 784 pixels")
    
    # Use subset for faster training (optional)
    num_train_samples = 20000  # Use 5000 instead of 60000 for faster training
    num_test_samples = 10000   # Use 1000 instead of 10000 for faster testing
    
    print(f"\nPreparing dataset (using {num_train_samples} training samples)...")
    
    # Prepare training data
    train_input, train_output = prepare_mnist_for_neural_net(
        x_train, y_train, num_train_samples
    )
    
    # Prepare test data
    test_input, test_output = prepare_mnist_for_neural_net(
        x_test, y_test, num_test_samples
    )
    
    # Show sample
    print("\nSample train data:")
    for i in range(3):
        digit = y_train[i]
        print(f"  Image {i}: Digit {digit} → One-hot: {train_output[i]}")
        print(f"    First 10 pixels: {[f'{p:.2f}' for p in train_input[i][:10]]}")
    
    # Count class distribution
    class_counts = [0] * 10
    for output in train_output:
        digit = output.index(1.0)
        class_counts[digit] += 1
    
    print(f"\nClass distribution (training):")
    for digit in range(10):
        print(f"  Digit {digit}: {class_counts[digit]} samples")
    
    # Save training data to data directory
    train_data = {
        "Input_Values": train_input,
        "Output_Values": train_output
    }
    
    data_dir = os.path.join(project_root, 'data')
    train_file = os.path.join(data_dir, "mnist_train.json")
    
    print(f"\nSaving training data to: {train_file}")
    with open(train_file, "w") as file:
        json.dump(train_data, file)
    
    # Save test data
    test_data = {
        "Input_Values": test_input,
        "Output_Values": test_output
    }
    
    test_file = os.path.join(data_dir, "mnist_test.json")
    print(f"Saving test data to: {test_file}")
    with open(test_file, "w") as file:
        json.dump(test_data, file)
    
    print("\n" + "=" * 70)
    print("✅ MNIST DATASET READY!")
    print("Data files: 'mnist_train.json' and 'mnist_test.json'")
    print("=" * 70)
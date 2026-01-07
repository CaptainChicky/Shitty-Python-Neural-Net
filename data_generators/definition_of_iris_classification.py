import json
import numpy as np
import os

# Task: Classify iris flowers into 3 species based on measurements

def generate_iris_data(num_samples=900):
    """
    Generate synthetic iris-like flower classification data in a 4D hypercube.

    Features (inputs):
    - Sepal length (normalized 0-1)
    - Sepal width (normalized 0-1)
    - Petal length (normalized 0-1)
    - Petal width (normalized 0-1)

    Classes (outputs are one-hot encoded):
    - Setosa: [1, 0, 0]
    - Versicolor: [0, 1, 0]
    - Virginica: [0, 0, 1]
    """

    input_data = []
    output_data = []

    samples_per_class = num_samples // 3

    # Class 0: Setosa (small petals, wide sepals)
    for _ in range(samples_per_class):
        sepal_length = np.random.uniform(0.2, 0.4)   # Smaller range
        sepal_width = np.random.uniform(0.6, 0.8)    # Wider range
        petal_length = np.random.uniform(0.1, 0.3)   # Small petals
        petal_width = np.random.uniform(0.05, 0.25)  # Narrow petals

        input_values = [
            float(sepal_length),
            float(sepal_width),
            float(petal_length),
            float(petal_width)
        ]

        # One-hot: Setosa = [1, 0, 0]
        output_values = [1.0, 0.0, 0.0]

        input_data.append(input_values)
        output_data.append(output_values)

    # Class 1: Versicolor (medium size)
    for _ in range(samples_per_class):
        sepal_length = np.random.uniform(0.5, 0.7)   # Medium range
        sepal_width = np.random.uniform(0.4, 0.6)    # Medium range
        petal_length = np.random.uniform(0.4, 0.6)   # Medium petals
        petal_width = np.random.uniform(0.4, 0.6)    # Medium petals

        input_values = [
            float(sepal_length),
            float(sepal_width),
            float(petal_length),
            float(petal_width)
        ]

        # One-hot: Versicolor = [0, 1, 0]
        output_values = [0.0, 1.0, 0.0]

        input_data.append(input_values)
        output_data.append(output_values)

    # Class 2: Virginica (large petals, narrow sepals)
    for _ in range(samples_per_class):
        sepal_length = np.random.uniform(0.7, 0.9)   # Larger range
        sepal_width = np.random.uniform(0.3, 0.5)    # Narrower range
        petal_length = np.random.uniform(0.7, 0.9)   # Large petals
        petal_width = np.random.uniform(0.7, 0.9)    # Wide petals

        input_values = [
            float(sepal_length),
            float(sepal_width),
            float(petal_length),
            float(petal_width)
        ]

        # One-hot: Virginica = [0, 0, 1]
        output_values = [0.0, 0.0, 1.0]

        input_data.append(input_values)
        output_data.append(output_values)

    # Shuffle the data
    combined = list(zip(input_data, output_data))
    np.random.shuffle(combined)
    input_data, output_data = zip(*combined)

    return list(input_data), list(output_data)


if __name__ == "__main__":
    print("=" * 70)
    print("Generating Iris Classification Data (3 Classes)")
    print("=" * 70)

    # Generate training data
    input_data, output_data = generate_iris_data(num_samples=900)

    print(f"\nGenerated {len(input_data)} samples")
    print(f"Input features: 4 (sepal length, sepal width, petal length, petal width)")
    print(f"Output classes: 3 (Setosa, Versicolor, Virginica)")
    print(f"Output format: one-hot encoded")
    print(f"\nSample data:")

    class_names = ["Setosa", "Versicolor", "Virginica"]
    for i in range(6):
        class_idx = output_data[i].index(1.0)
        print(f"  Sample {i+1}:")
        print(f"    Features: {[f'{v:.3f}' for v in input_data[i]]}")
        print(f"    Class: {class_names[class_idx]} {output_data[i]}")

    # Count samples per class
    class_counts = [0, 0, 0]
    for output in output_data:
        class_idx = output.index(1.0)
        class_counts[class_idx] += 1

    print(f"\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_counts[i]} samples")

    # Save to JSON
    data = {
        "Input_Values": input_data,
        "Output_Values": output_data
    }

    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, "iris_data.json")

    with open(output_file, "w") as file:
        json.dump(data, file)

    print(f"\nData saved to: {output_file}")
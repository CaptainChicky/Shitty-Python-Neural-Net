import json
import numpy as np
import os

# This script generates data for testing LINEAR ACTIVATION (regression with unbounded outputs)
# Task: Predict house prices based on features
# Formula: price = 150 * sqft + 50 * bedrooms + 30 * age + noise
# This simulates a real-world regression problem with unbounded continuous outputs

def generate_house_price_data(num_samples=1000):
    """
    Generate synthetic house price data for regression testing.

    Features (inputs):
    - Square footage (normalized 0-1, representing 500-3000 sqft)
    - Number of bedrooms (normalized 0-1, representing 1-5 bedrooms)
    - Age of house (normalized 0-1, representing 0-50 years)

    Target (output):
    - Price in $1000s (unbounded, typically 100-600)

    This tests LINEAR activation which allows unbounded outputs.
    """

    # Generate random features (normalized 0-1)
    sqft_normalized = np.random.uniform(0, 1, num_samples)      # 500-3000 sqft
    bedrooms_normalized = np.random.uniform(0, 1, num_samples)   # 1-5 bedrooms
    age_normalized = np.random.uniform(0, 1, num_samples)        # 0-50 years

    # Convert to actual values for price calculation
    sqft_actual = 500 + sqft_normalized * 2500              # 500 to 3000 sqft
    bedrooms_actual = 1 + bedrooms_normalized * 4           # 1 to 5 bedrooms
    age_actual = age_normalized * 50                        # 0 to 50 years

    # Calculate price with realistic formula (in $1000s)
    # Bigger houses cost more, more bedrooms cost more, older houses cost less
    base_price = 100  # $100k base
    price = (
        base_price +
        0.15 * sqft_actual +      # $150 per sqft
        50 * bedrooms_actual +     # $50k per bedroom
        -2 * age_actual +          # -$2k per year of age
        np.random.normal(0, 20, num_samples)  # +/- $20k noise
    )

    # Create input/output arrays
    input_data = []
    output_data = []

    for i in range(num_samples):
        # Inputs: normalized features
        input_values = [
            float(sqft_normalized[i]),
            float(bedrooms_normalized[i]),
            float(age_normalized[i])
        ]

        # Output: price in $1000s (unbounded!)
        output_value = [float(price[i])]

        input_data.append(input_values)
        output_data.append(output_value)

    return input_data, output_data


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING LINEAR REGRESSION DATA (House Prices)")
    print("=" * 70)

    # Generate training data
    input_data, output_data = generate_house_price_data(num_samples=1000)

    print(f"\nGenerated {len(input_data)} samples")
    print(f"Input features: 3 (sqft, bedrooms, age)")
    print(f"Output: 1 (price in $1000s)")
    print(f"\nSample data:")
    for i in range(5):
        print(f"  Sample {i+1}:")
        print(f"    Features (normalized): {input_data[i]}")
        print(f"    Price: ${output_data[i][0]:.2f}k")

    # Calculate statistics
    prices = [output_data[i][0] for i in range(len(output_data))]
    print(f"\nPrice statistics:")
    print(f"  Min:  ${min(prices):.2f}k")
    print(f"  Max:  ${max(prices):.2f}k")
    print(f"  Mean: ${np.mean(prices):.2f}k")
    print(f"  Std:  ${np.std(prices):.2f}k")

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

    output_file = os.path.join(data_dir, "linear_regression_data.json")

    with open(output_file, "w") as file:
        json.dump(data, file)

    print(f"\nData saved to: {output_file}")
    print("=" * 70)
    print("NOTE: This data tests LINEAR activation (unbounded outputs)")
    print("Use with: activation_func=ActivationFunction.linear")
    print("Use with: cost_function='mse' or 'mae'")
    print("=" * 70)
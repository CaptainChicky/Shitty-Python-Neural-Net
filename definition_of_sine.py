import json
import math
import random

# Sine Wave Classification
# Task: Given a point (x, y), determine if the point is ABOVE or BELOW a sine wave
# This tests if the network can learn periodic/oscillating patterns

def is_above_sine(x, y):
    """
    Check if point (x, y) is above the curve y = sin(x)
    Returns True if y > sin(x), False otherwise
    """
    sine_value = math.sin(x)
    return y > sine_value

def generate_output(is_above):
    """
    Output encoding:
    - Above sine wave → [1, -1]
    - Below sine wave → [-1, 1]
    """
    if is_above:
        return (1, -1)
    else:
        return (-1, 1)

# Generate balanced dataset
# We'll generate points in the range x ∈ [0, 2π], y ∈ [-1.5, 1.5]
# This covers one full sine wave period

samples_per_class = 250  # 250 above + 250 below = 500 total

above_samples = []
below_samples = []

print("Generating sine wave dataset...")
print(f"Target: {samples_per_class} above samples, {samples_per_class} below samples")
print("Domain: x ∈ [0, 2π], y ∈ [-1.5, 1.5]")

# Generate samples
max_attempts = 50000
attempts = 0

while (len(above_samples) < samples_per_class or len(below_samples) < samples_per_class) and attempts < max_attempts:
    # Random point in domain
    x = random.uniform(0, 2 * math.pi)
    y = random.uniform(-1.5, 1.5)
    attempts += 1

    # Classify
    if is_above_sine(x, y) and len(above_samples) < samples_per_class:
        above_samples.append((x, y))
    elif not is_above_sine(x, y) and len(below_samples) < samples_per_class:
        below_samples.append((x, y))

    # Progress update
    if attempts % 10000 == 0:
        print(f"  Attempt {attempts}: {len(above_samples)} above, {len(below_samples)} below")

print(f"\nGeneration complete after {attempts} attempts:")
print(f"  Above samples: {len(above_samples)}")
print(f"  Below samples: {len(below_samples)}")

# Combine and shuffle
all_samples = above_samples + below_samples
random.shuffle(all_samples)

# Create input and output data
data_entry_1 = [[x, y] for x, y in all_samples]
data_entry_2 = []

for x, y in all_samples:
    is_above = is_above_sine(x, y)
    data_entry_2.append(generate_output(is_above))

# Verify balance
num_above = sum(1 for output in data_entry_2 if output == (1, -1))
num_below = sum(1 for output in data_entry_2 if output == (-1, 1))
print(f"\nFinal dataset composition:")
print(f"  Above sine: {num_above} ({num_above/len(data_entry_2)*100:.1f}%)")
print(f"  Below sine: {num_below} ({num_below/len(data_entry_2)*100:.1f}%)")

# Save to JSON
data = {
    "Input_Values": data_entry_1,
    "Output_Values": data_entry_2
}

with open("sine_data.json", "w") as file:
    json.dump(data, file)

print("\nSaved to sine_data.json")
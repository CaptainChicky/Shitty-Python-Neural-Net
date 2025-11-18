import json
import random
import os

# XOR (Exclusive OR) - The Classic Neural Network Test
# This is historically significant because it proved that you NEED hidden layers
# A simple perceptron (no hidden layers) cannot learn XOR, but a network with hidden layers can

def xor_function(x1, x2):
    """
    XOR truth table:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0
    """
    return (x1 + x2) % 2  # XOR is equivalent to (x1 + x2) mod 2

def generate_output(result):
    """
    XOR result: 0 or 1
    We encode as:
    - 0 → [-1, 1]  (first output negative, second positive)
    - 1 → [1, -1]  (first output positive, second negative)
    """
    if result == 1:
        return (1, -1)
    else:
        return (-1, 1)

# Generate XOR dataset with noise
# The pure XOR problem only has 4 samples, which is too small for training
# So we add gaussian noise to create more samples while keeping the same pattern

samples_per_class = 250  # 250 zeros + 250 ones = 500 total
noise_std = 0.1  # Standard deviation of gaussian noise

xor_zero_samples = []  # XOR = 0 samples
xor_one_samples = []   # XOR = 1 samples

print("Generating XOR dataset with noise...")
print(f"Target: {samples_per_class} XOR=0 samples, {samples_per_class} XOR=1 samples")
print(f"Noise std dev: {noise_std}")

# Generate samples for XOR = 0 (from 0,0 and 1,1)
for _ in range(samples_per_class // 2):
    # Base: (0, 0) with noise
    x1 = 0 + random.gauss(0, noise_std)
    x2 = 0 + random.gauss(0, noise_std)
    xor_zero_samples.append((x1, x2))

for _ in range(samples_per_class // 2):
    # Base: (1, 1) with noise
    x1 = 1 + random.gauss(0, noise_std)
    x2 = 1 + random.gauss(0, noise_std)
    xor_zero_samples.append((x1, x2))

# Generate samples for XOR = 1 (from 0,1 and 1,0)
for _ in range(samples_per_class // 2):
    # Base: (0, 1) with noise
    x1 = 0 + random.gauss(0, noise_std)
    x2 = 1 + random.gauss(0, noise_std)
    xor_one_samples.append((x1, x2))

for _ in range(samples_per_class // 2):
    # Base: (1, 0) with noise
    x1 = 1 + random.gauss(0, noise_std)
    x2 = 0 + random.gauss(0, noise_std)
    xor_one_samples.append((x1, x2))

# Combine and shuffle
all_samples = xor_zero_samples + xor_one_samples
random.shuffle(all_samples)

print(f"\nGeneration complete:")
print(f"  XOR=0 samples: {len(xor_zero_samples)}")
print(f"  XOR=1 samples: {len(xor_one_samples)}")

# Create input and output data
data_entry_1 = [[x1, x2] for x1, x2 in all_samples]
data_entry_2 = []

for x1, x2 in all_samples:
    # Round to nearest integer for XOR calculation (to handle noise)
    x1_binary = 1 if x1 > 0.5 else 0
    x2_binary = 1 if x2 > 0.5 else 0
    xor_result = xor_function(x1_binary, x2_binary)
    data_entry_2.append(generate_output(xor_result))

# Verify balance
num_zeros = sum(1 for output in data_entry_2 if output == (-1, 1))
num_ones = sum(1 for output in data_entry_2 if output == (1, -1))
print(f"\nFinal dataset composition:")
print(f"  XOR=0: {num_zeros} ({num_zeros/len(data_entry_2)*100:.1f}%)")
print(f"  XOR=1: {num_ones} ({num_ones/len(data_entry_2)*100:.1f}%)")

# Save to JSON
data = {
    "Input_Values": data_entry_1,
    "Output_Values": data_entry_2
}

data_file = os.path.join(os.path.dirname(__file__), "..", "data", "xor_data.json")
with open(data_file, "w") as file:
    json.dump(data, file)

print(f"\nSaved to {data_file}")
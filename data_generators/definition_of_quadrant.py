import json
import random
import os

# Quadrant Classification - MULTI-CLASS (4 classes!)
# Task: Given a point (x, y), determine which quadrant it's in
# This is different from binary classification - we have 4 possible outputs!

# Quadrant definition:
# Q1: x > 0, y > 0  (top right)
# Q2: x < 0, y > 0  (top left)
# Q3: x < 0, y < 0  (bottom left)
# Q4: x > 0, y < 0  (bottom right)

def get_quadrant(x, y):
    """
    Returns quadrant number (1, 2, 3, or 4)
    We avoid points exactly on the axes by using a small threshold
    """
    threshold = 0.01

    # Ensure point is not too close to axes
    if abs(x) < threshold or abs(y) < threshold:
        # Nudge it slightly away from axis
        if abs(x) < threshold:
            x = threshold if random.random() > 0.5 else -threshold
        if abs(y) < threshold:
            y = threshold if random.random() > 0.5 else -threshold

    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:  # x > 0 and y < 0
        return 4

def generate_output(quadrant):
    """
    Multi-class output encoding (4 neurons instead of 2!)
    We use one-hot encoding:
    - Quadrant 1 → [1, -1, -1, -1]
    - Quadrant 2 → [-1, 1, -1, -1]
    - Quadrant 3 → [-1, -1, 1, -1]
    - Quadrant 4 → [-1, -1, -1, 1]

    The neuron that's "1" indicates which quadrant
    """
    if quadrant == 1:
        return (1, -1, -1, -1)
    elif quadrant == 2:
        return (-1, 1, -1, -1)
    elif quadrant == 3:
        return (-1, -1, 1, -1)
    else:  # quadrant == 4
        return (-1, -1, -1, 1)

# Generate balanced dataset
# We want equal samples from each quadrant (4 classes)

samples_per_quadrant = 250  # 250 * 4 = 1000 total

q1_samples = []
q2_samples = []
q3_samples = []
q4_samples = []

print("Generating quadrant dataset (MULTI-CLASS)...")
print(f"Target: {samples_per_quadrant} samples per quadrant")
print("Domain: x ∈ [-5, 5], y ∈ [-5, 5]")

# Generate samples
max_attempts = 10000
attempts = 0

while (len(q1_samples) < samples_per_quadrant or
       len(q2_samples) < samples_per_quadrant or
       len(q3_samples) < samples_per_quadrant or
       len(q4_samples) < samples_per_quadrant) and attempts < max_attempts:

    # Random point in domain (avoiding axes)
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    attempts += 1

    # Skip if too close to axes
    if abs(x) < 0.1 or abs(y) < 0.1:
        continue

    # Classify and add to appropriate list
    quad = get_quadrant(x, y)
    if quad == 1 and len(q1_samples) < samples_per_quadrant:
        q1_samples.append((x, y))
    elif quad == 2 and len(q2_samples) < samples_per_quadrant:
        q2_samples.append((x, y))
    elif quad == 3 and len(q3_samples) < samples_per_quadrant:
        q3_samples.append((x, y))
    elif quad == 4 and len(q4_samples) < samples_per_quadrant:
        q4_samples.append((x, y))

print(f"\nGeneration complete after {attempts} attempts:")
print(f"  Q1 samples: {len(q1_samples)}")
print(f"  Q2 samples: {len(q2_samples)}")
print(f"  Q3 samples: {len(q3_samples)}")
print(f"  Q4 samples: {len(q4_samples)}")

# Combine and shuffle
all_samples = q1_samples + q2_samples + q3_samples + q4_samples
random.shuffle(all_samples)

# Create input and output data
data_entry_1 = [[x, y] for x, y in all_samples]
data_entry_2 = []

for x, y in all_samples:
    quad = get_quadrant(x, y)
    data_entry_2.append(generate_output(quad))

# Verify balance
num_q1 = sum(1 for output in data_entry_2 if output == (1, -1, -1, -1))
num_q2 = sum(1 for output in data_entry_2 if output == (-1, 1, -1, -1))
num_q3 = sum(1 for output in data_entry_2 if output == (-1, -1, 1, -1))
num_q4 = sum(1 for output in data_entry_2 if output == (-1, -1, -1, 1))

print(f"\nFinal dataset composition:")
print(f"  Q1: {num_q1} ({num_q1/len(data_entry_2)*100:.1f}%)")
print(f"  Q2: {num_q2} ({num_q2/len(data_entry_2)*100:.1f}%)")
print(f"  Q3: {num_q3} ({num_q3/len(data_entry_2)*100:.1f}%)")
print(f"  Q4: {num_q4} ({num_q4/len(data_entry_2)*100:.1f}%)")

# Save to JSON
data = {
    "Input_Values": data_entry_1,
    "Output_Values": data_entry_2
}

data_file = os.path.join(os.path.dirname(__file__), "..", "data", "quadrant_data.json")
with open(data_file, "w") as file:
    json.dump(data, file)

print(f"\nSaved to {data_file}")
print("\nNOTE: This is a MULTI-CLASS problem (4 classes)")
print("Your network needs 4 OUTPUT neurons instead of 2!")
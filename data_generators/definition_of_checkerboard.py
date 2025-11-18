import json
import random
import os

# Checkerboard Pattern Classification
# Task: Given a point (x, y), determine if it's on a "black" or "white" square
# Like a chess board pattern - alternating squares

def is_black_square(x, y, grid_size=1.0):
    """
    Determine if point (x, y) is on a black square of a checkerboard

    The rule: If (floor(x/grid_size) + floor(y/grid_size)) is even → black
                                                           is odd  → white

    This creates an alternating pattern like a checkerboard
    """
    grid_x = int(x // grid_size)
    grid_y = int(y // grid_size)
    return (grid_x + grid_y) % 2 == 0

def generate_output(is_black):
    """
    Output encoding:
    - Black square → [1, -1]
    - White square → [-1, 1]
    """
    if is_black:
        return (1, -1)
    else:
        return (-1, 1)

# Generate balanced dataset
# Domain: x ∈ [0, 4], y ∈ [0, 4]  (4x4 grid = 16 squares total, 8 black, 8 white)

samples_per_class = 500  # 500 black + 500 white = 1000 total
grid_size = 1.0  # Size of each square

black_samples = []
white_samples = []

print("Generating checkerboard dataset...")
print(f"Target: {samples_per_class} black samples, {samples_per_class} white samples")
print(f"Domain: x ∈ [0, 4], y ∈ [0, 4]")
print(f"Grid size: {grid_size}")

# Generate samples
max_attempts = 10000
attempts = 0

while (len(black_samples) < samples_per_class or len(white_samples) < samples_per_class) and attempts < max_attempts:
    # Random point in domain
    x = random.uniform(0, 4)
    y = random.uniform(0, 4)
    attempts += 1

    # Classify
    if is_black_square(x, y, grid_size) and len(black_samples) < samples_per_class:
        black_samples.append((x, y))
    elif not is_black_square(x, y, grid_size) and len(white_samples) < samples_per_class:
        white_samples.append((x, y))

print(f"\nGeneration complete after {attempts} attempts:")
print(f"  Black samples: {len(black_samples)}")
print(f"  White samples: {len(white_samples)}")

# Combine and shuffle
all_samples = black_samples + white_samples
random.shuffle(all_samples)

# Create input and output data
data_entry_1 = [[x, y] for x, y in all_samples]
data_entry_2 = []

for x, y in all_samples:
    is_black = is_black_square(x, y, grid_size)
    data_entry_2.append(generate_output(is_black))

# Verify balance
num_black = sum(1 for output in data_entry_2 if output == (1, -1))
num_white = sum(1 for output in data_entry_2 if output == (-1, 1))
print(f"\nFinal dataset composition:")
print(f"  Black: {num_black} ({num_black/len(data_entry_2)*100:.1f}%)")
print(f"  White: {num_white} ({num_white/len(data_entry_2)*100:.1f}%)")

# Save to JSON
data = {
    "Input_Values": data_entry_1,
    "Output_Values": data_entry_2
}

data_file = os.path.join(os.path.dirname(__file__), "..", "data", "checkerboard_data.json")
with open(data_file, "w") as file:
    json.dump(data, file)

print(f"\nSaved to {data_file}")
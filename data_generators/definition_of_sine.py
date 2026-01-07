import json
import math
import random
import os

# Sine Wave Classification
# Task: Given a point (x, y), determine if the point is ABOVE or BELOW a sine wave

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
    - Above sine wave → [1]
    - Below sine wave → [0]
    """
    if is_above:
        return [1]
    else:
        return [0]

def generate_boundary_focused_sample(is_above_class, distance_from_sine):
    """
    Generate a sample near the sine wave boundary.
    
    Args:
        is_above_class: True for above, False for below
        distance_from_sine: How far from y = sin(x) (e.g., 0.1 = within ±0.1)
    
    Returns:
        (x, y) tuple near the sine boundary
    """
    max_attempts = 1000
    x_min, x_max = -2 * math.pi, 4 * math.pi
    
    for _ in range(max_attempts):
        # Random x in domain
        x = random.uniform(x_min, x_max)
        
        # Calculate sine value at this x
        sine_val = math.sin(x)
        
        # Generate y close to sine boundary
        if is_above_class:
            # Above: y slightly above sin(x)
            y = sine_val + random.uniform(0, distance_from_sine)
        else:
            # Below: y slightly below sin(x)
            y = sine_val - random.uniform(0, distance_from_sine)
        
        # Check if within valid y range
        if -1.5 <= y <= 1.5:
            # Verify it's in the correct class
            actual_is_above = is_above_sine(x, y)
            if actual_is_above == is_above_class:
                return (x, y)
    
    # Fallback to regular generation
    print(f"Warning: Boundary generation failed, using fallback")
    return generate_regular_sample(is_above_class)

def generate_regular_sample(is_above_class):
    """Generate a regular random sample in the specified class."""
    x_min, x_max = -2 * math.pi, 4 * math.pi
    y_min, y_max = -1.5, 1.5
    
    max_attempts = 1000
    for _ in range(max_attempts):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        if is_above_sine(x, y) == is_above_class:
            return (x, y)
    
    # Should never happen, but just in case
    return (0, 0)

############################################################################################################
# CONFIGURATION
############################################################################################################
samples_per_class_regular = 300  # Regular samples per class
samples_per_class_boundary = 500  # Boundary-focused samples per class
boundary_distance = 0.1  # Distance from sine curve for boundary samples

print("=" * 80)
print(" " * 20 + "Sine Wave Data Generation (3 Periods)")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Domain:                           x ∈ [-2π, 4π] (3 full periods)")
print(f"                                    y ∈ [-1.5, 1.5]")
print(f"  Phase 1 - Regular samples:        {samples_per_class_regular} per class")
print(f"  Phase 2 - Boundary samples:       {samples_per_class_boundary} per class")
print(f"  Boundary distance:                ±{boundary_distance} from y = sin(x)")
print("=" * 80)

above_samples = []
below_samples = []

############################################################################################################
# PHASE 1: Regular generation across 3 periods
############################################################################################################
print(f"\n{'='*80}")
print("PHASE 1: REGULAR SAMPLES (Random across 3 periods)")
print(f"{'='*80}")
print(f"Target: {samples_per_class_regular} above, {samples_per_class_regular} below")

phase1_start_above = len(above_samples)
phase1_start_below = len(below_samples)

# Generate regular samples
for i in range(samples_per_class_regular):
    above_samples.append(generate_regular_sample(is_above_class=True))
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{samples_per_class_regular} above samples")

for i in range(samples_per_class_regular):
    below_samples.append(generate_regular_sample(is_above_class=False))
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{samples_per_class_regular} below samples")

phase1_above_added = len(above_samples) - phase1_start_above
phase1_below_added = len(below_samples) - phase1_start_below

print(f"\n✅ Phase 1 Complete:")
print(f"   Above samples: {phase1_above_added}")
print(f"   Below samples: {phase1_below_added}")
print(f"   Total so far:  {len(above_samples) + len(below_samples)}")

# Show some examples
print(f"\n   Sample above points:")
for x, y in above_samples[:3]:
    sine_val = math.sin(x)
    dist = y - sine_val
    print(f"     ({x:.2f}, {y:.2f}) → sin(x)={sine_val:.2f}, distance above={dist:.2f}")

############################################################################################################
# PHASE 2: Boundary-focused generation
############################################################################################################
print(f"\n{'='*80}")
print("PHASE 2: BOUNDARY-FOCUSED SAMPLES (Dense near y = sin(x))")
print(f"{'='*80}")
print(f"Generating samples within ±{boundary_distance} of the sine curve")

phase2_start_above = len(above_samples)
phase2_start_below = len(below_samples)

# Generate above samples near boundary
print(f"\nGenerating {samples_per_class_boundary} ABOVE samples (y ∈ [sin(x), sin(x)+{boundary_distance}])...")
for i in range(samples_per_class_boundary):
    above_samples.append(generate_boundary_focused_sample(is_above_class=True, 
                                                          distance_from_sine=boundary_distance))
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{samples_per_class_boundary} boundary above samples")

# Generate below samples near boundary
print(f"\nGenerating {samples_per_class_boundary} BELOW samples (y ∈ [sin(x)-{boundary_distance}, sin(x)])...")
for i in range(samples_per_class_boundary):
    below_samples.append(generate_boundary_focused_sample(is_above_class=False, 
                                                          distance_from_sine=boundary_distance))
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{samples_per_class_boundary} boundary below samples")

phase2_above_added = len(above_samples) - phase2_start_above
phase2_below_added = len(below_samples) - phase2_start_below

print(f"\n✅ Phase 2 Complete:")
print(f"   Above samples: {phase2_above_added}")
print(f"   Below samples: {phase2_below_added}")
print(f"   Total:         {len(above_samples) + len(below_samples)}")

# Show boundary samples
print(f"\n   Sample boundary above points:")
for x, y in above_samples[-3:]:
    sine_val = math.sin(x)
    dist = y - sine_val
    print(f"     ({x:.2f}, {y:.2f}) → sin(x)={sine_val:.2f}, distance above={dist:.3f}")

############################################################################################################
# Combine and save
############################################################################################################
print(f"\n{'='*80}")
print("FINAL DATASET SUMMARY")
print(f"{'='*80}")

print(f"\nSamples by phase:")
print(f"  Phase 1 (Regular):    {phase1_above_added} above + {phase1_below_added} below = {phase1_above_added + phase1_below_added}")
print(f"  Phase 2 (Boundary):   {phase2_above_added} above + {phase2_below_added} below = {phase2_above_added + phase2_below_added}")
print(f"  {'─'*60}")
print(f"  TOTAL:                {len(above_samples)} above + {len(below_samples)} below = {len(above_samples) + len(below_samples)}")

# Domain coverage
all_x_above = [x for x, y in above_samples]
all_x_below = [x for x, y in below_samples]

print(f"\nDomain coverage (x-axis):")
print(f"  Training domain:  [{-2*math.pi:.2f}, {4*math.pi:.2f}] ({-2*math.pi:.2f} to {4*math.pi:.2f})")
print(f"  That's 3 periods: [-2π, 0] + [0, 2π] + [2π, 4π]")
print(f"  Above samples x:  min={min(all_x_above):.2f}, max={max(all_x_above):.2f}")
print(f"  Below samples x:  min={min(all_x_below):.2f}, max={max(all_x_below):.2f}")

# Combine and shuffle
all_samples = above_samples + below_samples
random.shuffle(all_samples)

# Create input and output data (NO NORMALIZATION)
data_entry_1 = []
data_entry_2 = []

for x, y in all_samples:
    # Keep raw coordinates
    data_entry_1.append([x, y])
    
    # Generate output
    is_above = is_above_sine(x, y)
    data_entry_2.append(generate_output(is_above))

# Verify balance
num_above = sum(1 for output in data_entry_2 if output == [1])
num_below = sum(1 for output in data_entry_2 if output == [0])

print(f"\nFinal verification:")
print(f"  Above sine: {num_above} ({num_above/len(data_entry_2)*100:.1f}%)")
print(f"  Below sine: {num_below} ({num_below/len(data_entry_2)*100:.1f}%)")
print(f"\nOutput format: [1] for above, [0] for below")

# Save to JSON
data = {
    "Input_Values": data_entry_1,
    "Output_Values": data_entry_2
}

data_file = os.path.join(os.path.dirname(__file__), "..", "data", "sine_data.json")
with open(data_file, "w") as file:
    json.dump(data, file)

print(f"\nSaved to: {data_file}")
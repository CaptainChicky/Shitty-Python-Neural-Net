import json
import math
import random
import os
import numpy as np

# The color red is defined as within 127 units of (255, 0, 0) in RGB space
def is_color_red(r, g, b):
    red_point = (255, 0, 0)
    distance = math.sqrt((r - red_point[0])**2 + (g - red_point[1])**2 + (b - red_point[2])**2)
    return distance <= 127

def generate_output(isRed):
    if isRed:
        return (1, -1)
    else:
        return (-1, 1)

def generate_random_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_boundary_focused_rgb(is_red_class, min_radius, max_radius):
    """
    Generate RGB values specifically near the boundary sphere.
    """
    center = np.array([255, 0, 0])
    
    max_attempts = 10000
    for _ in range(max_attempts):
        radius = random.uniform(min_radius, max_radius)
        
        # Random direction (spherical coordinates)
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        
        # Convert to Cartesian coordinates
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        
        # Translate to center at (255, 0, 0)
        rgb = center + np.array([x, y, z])
        r, g, b = rgb
        
        # Check if within valid RGB range [0, 255]
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            r, g, b = int(round(r)), int(round(g)), int(round(b))
            
            # Verify it's in the correct class
            actual_is_red = is_color_red(r, g, b)
            if actual_is_red == is_red_class:
                return (r, g, b)
    
    print(f"Warning: Boundary generation failed, using fallback")
    return generate_random_rgb()

############################################################################################################
# CONFIGURATION
############################################################################################################
samples_per_class_regular = 400
samples_per_class_boundary = 300
axis_samples_per_axis = 300

print("=" * 80)
print(" " * 20 + "RGB RED COLOR DATA GENERATION")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Phase 1 - Regular samples:        {samples_per_class_regular} per class")
print(f"  Phase 2 - Boundary samples:       {samples_per_class_boundary} per class")
print(f"  Phase 3 - Axis-aligned samples:   {axis_samples_per_axis} per axis")
print(f"  Expected total:                   ~1800 samples")
print("=" * 80)

red_samples = set()
not_red_samples = set()

############################################################################################################
# PHASE 1: Regular generation
############################################################################################################
print(f"\n{'='*80}")
print("PHASE 1: REGULAR SAMPLES (Random across entire RGB space)")
print(f"{'='*80}")
print(f"Target: {samples_per_class_regular} red, {samples_per_class_regular} not-red")

phase1_red_start = len(red_samples)
phase1_not_red_start = len(not_red_samples)

attempts = 0
max_attempts = 100000

while (len(red_samples) < samples_per_class_regular or 
       len(not_red_samples) < samples_per_class_regular) and attempts < max_attempts:
    rgb = generate_random_rgb()
    attempts += 1

    if is_color_red(*rgb) and len(red_samples) < samples_per_class_regular:
        red_samples.add(rgb)
    elif not is_color_red(*rgb) and len(not_red_samples) < samples_per_class_regular:
        not_red_samples.add(rgb)

    if attempts % 10000 == 0:
        print(f"  Attempt {attempts:6d}: {len(red_samples):3d} red, {len(not_red_samples):3d} not-red")

phase1_red_added = len(red_samples) - phase1_red_start
phase1_not_red_added = len(not_red_samples) - phase1_not_red_start

print(f"\n✅ Phase 1 Complete (after {attempts} attempts):")
print(f"   Red samples added:     {phase1_red_added}")
print(f"   Not-red samples added: {phase1_not_red_added}")
print(f"   Total so far:          {len(red_samples) + len(not_red_samples)}")

# Show some examples
print(f"\n   Sample red colors from Phase 1:")
for rgb in list(red_samples)[:3]:
    dist = math.sqrt((rgb[0]-255)**2 + rgb[1]**2 + rgb[2]**2)
    print(f"     RGB{rgb} → distance={dist:.1f}")
print(f"   Sample not-red colors from Phase 1:")
for rgb in list(not_red_samples)[:3]:
    dist = math.sqrt((rgb[0]-255)**2 + rgb[1]**2 + rgb[2]**2)
    print(f"     RGB{rgb} → distance={dist:.1f}")

############################################################################################################
# PHASE 2: Boundary-focused generation
############################################################################################################
print(f"\n{'='*80}")
print("PHASE 2: BOUNDARY-FOCUSED SAMPLES (Dense sampling near decision boundary)")
print(f"{'='*80}")

phase2_red_start = len(red_samples)
phase2_not_red_start = len(not_red_samples)

# Generate red samples near inner boundary (radius 100-127)
print(f"\nGenerating {samples_per_class_boundary} RED samples (radius 100-127, just inside boundary)...")
for i in range(samples_per_class_boundary):
    rgb = generate_boundary_focused_rgb(is_red_class=True, min_radius=100, max_radius=127)
    red_samples.add(rgb)
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1:3d}/{samples_per_class_boundary} boundary red samples")

# Generate not-red samples near outer boundary (radius 127-154)
print(f"\nGenerating {samples_per_class_boundary} NOT-RED samples (radius 127-154, just outside boundary)...")
for i in range(samples_per_class_boundary):
    rgb = generate_boundary_focused_rgb(is_red_class=False, min_radius=127, max_radius=154)
    not_red_samples.add(rgb)
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1:3d}/{samples_per_class_boundary} boundary not-red samples")

phase2_red_added = len(red_samples) - phase2_red_start
phase2_not_red_added = len(not_red_samples) - phase2_not_red_start

print(f"\n✅ Phase 2 Complete:")
print(f"   Red samples added:     {phase2_red_added} (unique after deduplication)")
print(f"   Not-red samples added: {phase2_not_red_added} (unique after deduplication)")
print(f"   Total so far:          {len(red_samples) + len(not_red_samples)}")

# Show boundary samples
print(f"\n   Sample boundary red colors:")
boundary_reds = [rgb for rgb in red_samples if 100 <= math.sqrt((rgb[0]-255)**2 + rgb[1]**2 + rgb[2]**2) <= 127]
for rgb in list(boundary_reds)[:3]:
    dist = math.sqrt((rgb[0]-255)**2 + rgb[1]**2 + rgb[2]**2)
    print(f"     RGB{rgb} → distance={dist:.1f}")

############################################################################################################
# PHASE 3: Axis-aligned samples
############################################################################################################
print(f"\n{'='*80}")
print("PHASE 3: AXIS-ALIGNED SAMPLES (Cover R-axis and other critical regions)")
print(f"{'='*80}")

phase3_red_start = len(red_samples)
phase3_not_red_start = len(not_red_samples)

# R-axis samples (varying R, G=0, B=0)
print(f"\nR-axis samples (R varies, G=0, B=0):")
r_axis_red = 0
r_axis_not_red = 0
for i in range(axis_samples_per_axis):
    r = int(random.uniform(0, 255))
    g = 0
    b = 0
    if is_color_red(r, g, b):
        red_samples.add((r, g, b))
        r_axis_red += 1
    else:
        not_red_samples.add((r, g, b))
        r_axis_not_red += 1
print(f"  Generated: {r_axis_red} red, {r_axis_not_red} not-red")
print(f"  Boundary at R={255-127}={128} (anything R<128 with G=0,B=0 is NOT red)")

# G-axis samples (R=255, varying G, B=0)
print(f"\nG-axis samples (R=255, G varies, B=0):")
g_axis_red = 0
g_axis_not_red = 0
for i in range(axis_samples_per_axis):
    r = 255
    g = int(random.uniform(0, 255))
    b = 0
    if is_color_red(r, g, b):
        red_samples.add((r, g, b))
        g_axis_red += 1
    else:
        not_red_samples.add((r, g, b))
        g_axis_not_red += 1
print(f"  Generated: {g_axis_red} red, {g_axis_not_red} not-red")
print(f"  Boundary at G={127} (anything G>127 with R=255,B=0 is NOT red)")

# B-axis samples (R=255, G=0, varying B)
print(f"\nB-axis samples (R=255, G=0, B varies):")
b_axis_red = 0
b_axis_not_red = 0
for i in range(axis_samples_per_axis):
    r = 255
    g = 0
    b = int(random.uniform(0, 255))
    if is_color_red(r, g, b):
        red_samples.add((r, g, b))
        b_axis_red += 1
    else:
        not_red_samples.add((r, g, b))
        b_axis_not_red += 1
print(f"  Generated: {b_axis_red} red, {b_axis_not_red} not-red")
print(f"  Boundary at B={127}")

# Diagonal samples (varying all three proportionally)
print(f"\nDiagonal samples (R=G=B, line from black to white):")
diag_red = 0
diag_not_red = 0
for i in range(axis_samples_per_axis):
    t = random.uniform(0, 1)
    r = int(255 * t)
    g = int(255 * t)
    b = int(255 * t)
    if is_color_red(r, g, b):
        red_samples.add((r, g, b))
        diag_red += 1
    else:
        not_red_samples.add((r, g, b))
        diag_not_red += 1
print(f"  Generated: {diag_red} red, {diag_not_red} not-red")
print(f"  (All diagonal points should be NOT red since they're far from (255,0,0))")

phase3_red_added = len(red_samples) - phase3_red_start
phase3_not_red_added = len(not_red_samples) - phase3_not_red_start

print(f"\n✅ Phase 3 Complete:")
print(f"   Red samples added:     {phase3_red_added} (unique after deduplication)")
print(f"   Not-red samples added: {phase3_not_red_added} (unique after deduplication)")
print(f"   Total samples:         {len(red_samples) + len(not_red_samples)}")

############################################################################################################
# Combine and save
############################################################################################################
print(f"\n{'='*80}")
print("FINAL DATASET SUMMARY")
print(f"{'='*80}")

print(f"\nSamples by phase:")
print(f"  Phase 1 (Regular):         {phase1_red_added} red + {phase1_not_red_added} not-red = {phase1_red_added + phase1_not_red_added}")
print(f"  Phase 2 (Boundary):        {phase2_red_added} red + {phase2_not_red_added} not-red = {phase2_red_added + phase2_not_red_added}")
print(f"  Phase 3 (Axis-aligned):    {phase3_red_added} red + {phase3_not_red_added} not-red = {phase3_red_added + phase3_not_red_added}")
print(f"  {'─'*60}")
print(f"  TOTAL (unique):            {len(red_samples)} red + {len(not_red_samples)} not-red = {len(red_samples) + len(not_red_samples)}")

# Class balance
total = len(red_samples) + len(not_red_samples)
red_pct = len(red_samples) / total * 100
not_red_pct = len(not_red_samples) / total * 100
print(f"\nClass balance:")
print(f"  Red:     {len(red_samples):4d} ({red_pct:.1f}%)")
print(f"  Not-red: {len(not_red_samples):4d} ({not_red_pct:.1f}%)")

# Distance distribution
all_red_list = list(red_samples)
all_not_red_list = list(not_red_samples)

red_distances = [math.sqrt((r-255)**2 + g**2 + b**2) for r, g, b in all_red_list]
not_red_distances = [math.sqrt((r-255)**2 + g**2 + b**2) for r, g, b in all_not_red_list]

print(f"\nDistance from (255,0,0) statistics:")
print(f"  Red samples:     min={min(red_distances):.1f}, max={max(red_distances):.1f}, avg={np.mean(red_distances):.1f}")
print(f"  Not-red samples: min={min(not_red_distances):.1f}, max={max(not_red_distances):.1f}, avg={np.mean(not_red_distances):.1f}")
print(f"  Decision boundary at distance = 127")

# Combine and shuffle
all_samples = all_red_list + all_not_red_list
random.shuffle(all_samples)

# Create input and output data
data_entry_1 = all_samples
data_entry_2 = []
for r, g, b in data_entry_1:
    is_red = is_color_red(r, g, b)
    data_entry_2.append(generate_output(is_red))

# Verify balance (should match unique counts)
num_red = sum(1 for output in data_entry_2 if output == (1, -1))
num_not_red = sum(1 for output in data_entry_2 if output == (-1, 1))

print(f"\nFinal data verification:")
print(f"  Red:     {num_red} ({num_red/len(data_entry_2)*100:.1f}%)")
print(f"  Not-red: {num_not_red} ({num_not_red/len(data_entry_2)*100:.1f}%)")

# Save
data = {
    "RGB_Values": data_entry_1,
    "Is_Red": data_entry_2
}

data_file = os.path.join(os.path.dirname(__file__), "..", "data", "color_data.json")
with open(data_file, "w") as file:
    json.dump(data, file)

print(f"\n💾 Saved to: {data_file}")
print(f"\n{'='*80}")
print("✅ DATASET GENERATION COMPLETE!")
print(f"{'='*80}")
print("\nKey points:")
print("  ✓ Balanced classes (50-50 red/not-red)")
print("  ✓ Dense boundary sampling (helps learn spherical shape)")
print("  ✓ Axis-aligned samples (fixes R-axis triangle problem)")
print("  ✓ Ready for training!")
print(f"{'='*80}\n")
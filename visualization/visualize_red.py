import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_red.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'color_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['RGB_Values'])
train_outputs = np.array(data['Is_Red'])

# Create custom colormap: gray (not red) -> white -> salmon (red)
colors_list = ['#d3d3d3', '#ffffff', '#fa8072']  # lightgray -> white -> salmon
red_cmap = LinearSegmentedColormap.from_list('red_custom', colors_list, N=256)

# True red definition (for reference circles)
def is_actually_red(r, g, b):
    distance = np.sqrt((r - 255)**2 + (g - 0)**2 + (b - 0)**2)
    return distance <= 127

# Create figure with 4 subplots (2x2 grid of slices)
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

resolution = 500  # High resolution for smooth visualization

# ============================================
# Slice 1: R vs G (at B=0)
# ============================================
ax = axes[0, 0]
r_vals = np.linspace(0, 255, resolution)
g_vals = np.linspace(0, 255, resolution)
R, G = np.meshgrid(r_vals, g_vals)

predictions_raw = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([R[i, j], G[i, j], 0])  # B=0
        output = neural_net.forward_propagation(point)
        # Store raw tanh output (-1 to 1)
        predictions_raw[i, j] = output[0]

print(f"Slice 1 (B=0) - Raw predictions range: [{predictions_raw.min():.3f}, {predictions_raw.max():.3f}]")
print(f"  Values near 0 (uncertain): {np.sum(np.abs(predictions_raw) < 0.1)} / {resolution*resolution}")

# Plot with gradient
im = ax.imshow(predictions_raw, extent=[0, 255, 0, 255], origin='lower',
               cmap=red_cmap, vmin=-1, vmax=1, aspect='auto')
ax.contour(R, G, predictions_raw, levels=[0], colors='red', linewidths=2, linestyles='dotted')

# Draw true boundary circle
theta = np.linspace(0, 2*np.pi, 100)
# Circle equation: (r-255)^2 + (g-0)^2 = 127^2, with b=0
r_circle = 255 + 127 * np.cos(theta)
g_circle = 0 + 127 * np.sin(theta)
# Only draw where both r and g are in valid range [0, 255]
mask = (r_circle >= 0) & (r_circle <= 255) & (g_circle >= 0) & (g_circle <= 255)
ax.plot(r_circle[mask], g_circle[mask], 'b--', linewidth=2, label='True boundary')

# Overlay training data at this slice (B ≈ 0)
red_mask = np.array([output[0] == 1 for output in train_outputs])
slice_mask = np.abs(train_inputs[:, 2] - 0) < 20  # Within ±20 of B=0
ax.scatter(train_inputs[slice_mask & red_mask, 0], 
          train_inputs[slice_mask & red_mask, 1],
          c='red', s=10, alpha=0.6, edgecolors='darkred', linewidths=0.5)
ax.scatter(train_inputs[slice_mask & ~red_mask, 0], 
          train_inputs[slice_mask & ~red_mask, 1],
          c='gray', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)

ax.set_xlabel('Red (R)', fontsize=12, fontweight='bold')
ax.set_ylabel('Green (G)', fontsize=12, fontweight='bold')
ax.set_title('Slice at Blue = 0', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.grid(True, alpha=0.3)

# ============================================
# Slice 2: R vs B (at G=0)
# ============================================
ax = axes[0, 1]
r_vals = np.linspace(0, 255, resolution)
b_vals = np.linspace(0, 255, resolution)
R, B = np.meshgrid(r_vals, b_vals)

predictions_raw = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([R[i, j], 0, B[i, j]])  # G=0
        output = neural_net.forward_propagation(point)
        predictions_raw[i, j] = output[0]

print(f"Slice 2 (G=0) - Raw predictions range: [{predictions_raw.min():.3f}, {predictions_raw.max():.3f}]")
print(f"  Values near 0 (uncertain): {np.sum(np.abs(predictions_raw) < 0.1)} / {resolution*resolution}")

# Plot with gradient
im = ax.imshow(predictions_raw, extent=[0, 255, 0, 255], origin='lower',
               cmap=red_cmap, vmin=-1, vmax=1, aspect='auto')
ax.contour(R, B, predictions_raw, levels=[0], colors='red', linewidths=2, linestyles='dotted')

# Draw true boundary circle
r_circle = 255 + 127 * np.cos(theta)
b_circle = 0 + 127 * np.sin(theta)
mask = (r_circle >= 0) & (r_circle <= 255) & (b_circle >= 0) & (b_circle <= 255)
ax.plot(r_circle[mask], b_circle[mask], 'b--', linewidth=2, label='True boundary')

# Overlay training data at this slice (G ≈ 0)
slice_mask = np.abs(train_inputs[:, 1] - 0) < 20
ax.scatter(train_inputs[slice_mask & red_mask, 0], 
          train_inputs[slice_mask & red_mask, 2],
          c='red', s=10, alpha=0.6, edgecolors='darkred', linewidths=0.5)
ax.scatter(train_inputs[slice_mask & ~red_mask, 0], 
          train_inputs[slice_mask & ~red_mask, 2],
          c='gray', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)

ax.set_xlabel('Red (R)', fontsize=12, fontweight='bold')
ax.set_ylabel('Blue (B)', fontsize=12, fontweight='bold')
ax.set_title('Slice at Green = 0', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.grid(True, alpha=0.3)

# ============================================
# Slice 3: G vs B (at R=255)
# ============================================
ax = axes[1, 0]
g_vals = np.linspace(0, 255, resolution)
b_vals = np.linspace(0, 255, resolution)
G, B = np.meshgrid(g_vals, b_vals)

predictions_raw = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([255, G[i, j], B[i, j]])  # R=255
        output = neural_net.forward_propagation(point)
        predictions_raw[i, j] = output[0]

print(f"Slice 3 (R=255) - Raw predictions range: [{predictions_raw.min():.3f}, {predictions_raw.max():.3f}]")
print(f"  Values near 0 (uncertain): {np.sum(np.abs(predictions_raw) < 0.1)} / {resolution*resolution}")

# Plot with gradient
im = ax.imshow(predictions_raw, extent=[0, 255, 0, 255], origin='lower',
               cmap=red_cmap, vmin=-1, vmax=1, aspect='auto')
ax.contour(G, B, predictions_raw, levels=[0], colors='red', linewidths=2, linestyles='dotted')

# Draw true boundary circle (should be centered at G=0, B=0)
g_circle = 0 + 127 * np.cos(theta)
b_circle = 0 + 127 * np.sin(theta)
mask = (g_circle >= 0) & (g_circle <= 255) & (b_circle >= 0) & (b_circle <= 255)
ax.plot(g_circle[mask], b_circle[mask], 'b--', linewidth=2, label='True boundary')

# Overlay training data at this slice (R ≈ 255)
slice_mask = np.abs(train_inputs[:, 0] - 255) < 20
ax.scatter(train_inputs[slice_mask & red_mask, 1], 
          train_inputs[slice_mask & red_mask, 2],
          c='red', s=10, alpha=0.6, edgecolors='darkred', linewidths=0.5)
ax.scatter(train_inputs[slice_mask & ~red_mask, 1], 
          train_inputs[slice_mask & ~red_mask, 2],
          c='gray', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)

ax.set_xlabel('Green (G)', fontsize=12, fontweight='bold')
ax.set_ylabel('Blue (B)', fontsize=12, fontweight='bold')
ax.set_title('Slice at Red = 255', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.grid(True, alpha=0.3)

# ============================================
# Slice 4: R vs G (at B=127) - Different depth
# ============================================
ax = axes[1, 1]
r_vals = np.linspace(0, 255, resolution)
g_vals = np.linspace(0, 255, resolution)
R, G = np.meshgrid(r_vals, g_vals)

predictions_raw = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([R[i, j], G[i, j], 127])  # B=127
        output = neural_net.forward_propagation(point)
        predictions_raw[i, j] = output[0]

print(f"Slice 4 (B=127) - Raw predictions range: [{predictions_raw.min():.3f}, {predictions_raw.max():.3f}]")
print(f"  Values near 0 (uncertain): {np.sum(np.abs(predictions_raw) < 0.1)} / {resolution*resolution}")

# Plot with gradient
im = ax.imshow(predictions_raw, extent=[0, 255, 0, 255], origin='lower',
               cmap=red_cmap, vmin=-1, vmax=1, aspect='auto')
ax.contour(R, G, predictions_raw, levels=[0], colors='red', linewidths=2, linestyles='dotted')

# Draw true boundary circle
# For B=127: (r-255)^2 + (g-0)^2 + (127-0)^2 = 127^2
# So: (r-255)^2 + g^2 = 127^2 - 127^2 = 0
# This means only the point (255, 0, 127) is on the boundary!
# The circle has radius 0, so just mark the point
ax.plot(255, 0, 'bo', markersize=10, label='True boundary (single point)')

# Overlay training data at this slice (B ≈ 127)
slice_mask = np.abs(train_inputs[:, 2] - 127) < 20
ax.scatter(train_inputs[slice_mask & red_mask, 0], 
          train_inputs[slice_mask & red_mask, 1],
          c='red', s=10, alpha=0.6, edgecolors='darkred', linewidths=0.5)
ax.scatter(train_inputs[slice_mask & ~red_mask, 0], 
          train_inputs[slice_mask & ~red_mask, 1],
          c='gray', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)

ax.set_xlabel('Red (R)', fontsize=12, fontweight='bold')
ax.set_ylabel('Green (G)', fontsize=12, fontweight='bold')
ax.set_title('Slice at Blue = 127 (Edge of sphere)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.grid(True, alpha=0.3)

plt.suptitle('RGB Red Color Classification - 2D Slices Through Color Space\n' + 
             'Salmon = Model predicts RED | Gray = Model predicts NOT RED | Blue dashed = True boundary',
             fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'red_color_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to: {output_path}")

# Calculate accuracy
correct = 0
for i in range(len(train_inputs)):
    output = neural_net.forward_propagation(train_inputs[i])
    predicted_red = output[0] > 0  # Tanh: positive = red
    actual_red = train_outputs[i][0] == 1
    if predicted_red == actual_red:
        correct += 1

accuracy = (correct / len(train_inputs)) * 100
print(f"Training Accuracy: {accuracy:.2f}%")
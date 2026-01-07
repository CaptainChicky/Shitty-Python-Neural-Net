import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_checkerboard.json')
neural_net.load(model_path)

# Create a grid of points to visualize
resolution = 500  # Higher = smoother visualization
x = np.linspace(0, 4, resolution)
y = np.linspace(0, 4, resolution)
X, Y = np.meshgrid(x, y)

# Predict for each point - keep raw outputs
predictions_raw = np.zeros((resolution, resolution))
predictions_binary = np.zeros((resolution, resolution))

for i in range(resolution):
    for j in range(resolution):
        point = np.array([X[i, j], Y[i, j]])
        output = neural_net.forward_propagation(point)
        
        # Raw confidence (-1 to 1 for tanh)
        predictions_raw[i, j] = output[0]
        
        # Binary for accuracy
        predictions_binary[i, j] = 1 if output[0] > 0 else -1

# Create the actual checkerboard pattern
actual = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        grid_x = int(X[i, j] // 1.0)
        grid_y = int(Y[i, j] // 1.0)
        actual[i, j] = 1 if (grid_x + grid_y) % 2 == 0 else -1

# Calculate error based on raw confidence
confidence_error = np.abs(predictions_raw - actual)

# Plot side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# ============================================
# PLOT 1: Model's Prediction (RAW CONFIDENCE)
# ============================================
im1 = ax1.imshow(predictions_raw, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('Model Prediction (Confidence)\nBlack (+1) → White (-1)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)

# Add grid at square boundaries
for i in range(5):
    ax1.axhline(y=i, color='black', linewidth=1, alpha=0.4)
    ax1.axvline(x=i, color='black', linewidth=1, alpha=0.4)

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# ============================================
# PLOT 2: Actual Checkerboard
# ============================================
im2 = ax2.imshow(actual, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='RdBu_r', vmin=-1, vmax=1)
ax2.set_title('Actual Checkerboard', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('X', fontsize=11)
ax2.set_ylabel('Y', fontsize=11)

# Add grid
for i in range(5):
    ax2.axhline(y=i, color='black', linewidth=1, alpha=0.4)
    ax2.axvline(x=i, color='black', linewidth=1, alpha=0.4)

# Add invisible colorbar to maintain consistent sizing with plots 1 and 3
from matplotlib import cm
dummy_cbar = plt.colorbar(cm.ScalarMappable(cmap='RdBu_r'), ax=ax2, fraction=0.046, pad=0.04)
dummy_cbar.ax.set_visible(False)

# ============================================
# PLOT 3: Error Map (BASED ON CONFIDENCE)
# ============================================
im3 = ax3.imshow(confidence_error, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='YlOrRd', vmin=0, vmax=2)
ax3.set_title('Prediction Error (Confidence Distance)\nError: 0 (Perfect) → 2 (Wrong)', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('X', fontsize=11)
ax3.set_ylabel('Y', fontsize=11)

# Add grid
for i in range(5):
    ax3.axhline(y=i, color='gray', linewidth=1, alpha=0.3)
    ax3.axvline(x=i, color='gray', linewidth=1, alpha=0.3)

cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.suptitle('Checkerboard Classification - Confidence-Based Visualization', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'checkerboard_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Calculate statistics
accuracy = np.mean(predictions_binary == actual) * 100
mean_confidence = np.mean(np.abs(predictions_raw))
mean_error = np.mean(confidence_error)

print(f"\n{'='*60}")
print(f"Model Accuracy (Binary):     {accuracy:.2f}%")
print(f"Mean Confidence:             {mean_confidence:.3f} (0 to 1, higher = more confident)")
print(f"Mean Prediction Error:       {mean_error:.3f} (lower = better)")
print(f"Binary Errors:               {np.sum(predictions_binary != actual):,} / {resolution * resolution:,} points")
print(f"{'='*60}")

# Confidence breakdown
very_confident = np.sum(np.abs(predictions_raw) > 0.9)
confident = np.sum((np.abs(predictions_raw) > 0.7) & (np.abs(predictions_raw) <= 0.9))
uncertain = np.sum((np.abs(predictions_raw) > 0.3) & (np.abs(predictions_raw) <= 0.7))
very_uncertain = np.sum(np.abs(predictions_raw) <= 0.3)

total_points = resolution * resolution
print(f"\nConfidence Breakdown:")
print(f"  Very Confident (|out| > 0.9):  {very_confident:,} ({very_confident/total_points*100:.1f}%)")
print(f"  Confident (0.7 - 0.9):         {confident:,} ({confident/total_points*100:.1f}%)")
print(f"  Uncertain (0.3 - 0.7):         {uncertain:,} ({uncertain/total_points*100:.1f}%)")
print(f"  Very Uncertain (|out| < 0.3):  {very_uncertain:,} ({very_uncertain/total_points*100:.1f}%)")
print(f"{'='*60}")
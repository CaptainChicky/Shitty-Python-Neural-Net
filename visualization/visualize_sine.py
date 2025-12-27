import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_sine.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sine_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['Input_Values'])
train_outputs = np.array(data['Output_Values'])

# Separate above and below samples
above_mask = np.array([output[0] == 1 for output in train_outputs])
above_samples = train_inputs[above_mask]
below_samples = train_inputs[~above_mask]

# Extended range for bottom row
x_min_ext, x_max_ext = -4*np.pi, 6*np.pi
y_min_ext, y_max_ext = -4.0, 4.0

# Training region for top row
x_min_train, x_max_train = 0, 2*np.pi
y_min_train, y_max_train = -1.5, 1.5

print("Generating predictions for BOTH ranges...")

# Generate predictions for extended range
resolution_x_ext = 1200  
resolution_y_ext = 400

x_ext = np.linspace(x_min_ext, x_max_ext, resolution_x_ext)
y_ext = np.linspace(y_min_ext, y_max_ext, resolution_y_ext)
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

predictions_ext = np.zeros((resolution_y_ext, resolution_x_ext))
actual_ext = np.zeros((resolution_y_ext, resolution_x_ext))

print("  Extended range predictions...")
for i in range(resolution_y_ext):
    if i % 50 == 0:
        print(f"    Row {i}/{resolution_y_ext}...")
    for j in range(resolution_x_ext):
        point = np.array([X_ext[i, j], Y_ext[i, j]])
        output = neural_net.forward_propagation(point)
        predictions_ext[i, j] = 1 if output[0] > output[1] else -1
        sine_value = np.sin(X_ext[i, j])
        actual_ext[i, j] = 1 if Y_ext[i, j] > sine_value else -1

# Generate predictions for training region
resolution_x_train = 600
resolution_y_train = 250

x_train = np.linspace(x_min_train, x_max_train, resolution_x_train)
y_train = np.linspace(y_min_train, y_max_train, resolution_y_train)
X_train, Y_train = np.meshgrid(x_train, y_train)

predictions_train = np.zeros((resolution_y_train, resolution_x_train))
actual_train = np.zeros((resolution_y_train, resolution_x_train))

print("  Training region predictions (high detail)...")
for i in range(resolution_y_train):
    for j in range(resolution_x_train):
        point = np.array([X_train[i, j], Y_train[i, j]])
        output = neural_net.forward_propagation(point)
        predictions_train[i, j] = 1 if output[0] > output[1] else -1
        sine_value = np.sin(X_train[i, j])
        actual_train[i, j] = 1 if Y_train[i, j] > sine_value else -1

print("Rendering visualization...")

# Create figure with 2x3 grid (6 plots)
fig = plt.figure(figsize=(36, 20))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

# ============================================
# TOP ROW: TRAINING REGION (Zoomed In)
# ============================================

# TOP LEFT: Model Prediction (Training Region)
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(X_train, Y_train, predictions_train, levels=[-1.5, 0, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.6)
ax1.contour(X_train, Y_train, predictions_train, levels=[0], colors='red', linewidths=2)
ax1.scatter(above_samples[:, 0], above_samples[:, 1], c='blue', s=12, alpha=0.6, 
           edgecolors='darkblue', linewidths=0.5)
ax1.scatter(below_samples[:, 0], below_samples[:, 1], c='red', s=12, alpha=0.6,
           edgecolors='darkred', linewidths=0.5)
ax1.set_title('Model Prediction\n(Training Region)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min_train, x_max_train)
ax1.set_ylim(y_min_train, y_max_train)
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# TOP MIDDLE: Actual (Training Region)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(X_train, Y_train, actual_train, levels=[-1.5, 0, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.6)
x_sine_train = np.linspace(x_min_train, x_max_train, 500)
y_sine_train = np.sin(x_sine_train)
ax2.plot(x_sine_train, y_sine_train, 'black', linewidth=2, label='y = sin(x)')
ax2.fill_between(x_sine_train, y_sine_train, y_max_train, alpha=0.3, color='blue')
ax2.fill_between(x_sine_train, y_sine_train, y_min_train, alpha=0.3, color='red')
ax2.set_title('Actual Sine Wave\n(Training Region)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(x_min_train, x_max_train)
ax2.set_ylim(y_min_train, y_max_train)
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# TOP RIGHT: Errors (Training Region)
ax3 = fig.add_subplot(gs[0, 2])
difference_train = (predictions_train != actual_train).astype(float)
im3 = ax3.contourf(X_train, Y_train, difference_train, levels=[0, 0.5, 1], 
                   colors=['#90EE90', '#FF6B6B'], alpha=0.7)
ax3.contour(X_train, Y_train, predictions_train, levels=[0], colors='red', linewidths=1, linestyles='--')
ax3.plot(x_sine_train, y_sine_train, 'blue', linewidth=1, linestyle='--')
ax3.set_title('Errors\n(Training Region)', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(x_min_train, x_max_train)
ax3.set_ylim(y_min_train, y_max_train)
ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax3.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#90EE90', label='Correct'),
    Patch(facecolor='#FF6B6B', label='Error')
]
ax3.legend(handles=legend_elements, fontsize=10, loc='upper right')

# ============================================
# BOTTOM ROW: EXTENDED RANGE (Generalization)
# ============================================

# BOTTOM LEFT: Model Prediction (Extended)
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.contourf(X_ext, Y_ext, predictions_ext, levels=[-1.5, 0, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.6)
ax4.scatter(above_samples[:, 0], above_samples[:, 1], c='blue', s=6, alpha=0.7,
           edgecolors='darkblue', linewidths=0.4, zorder=5)
ax4.scatter(below_samples[:, 0], below_samples[:, 1], c='red', s=6, alpha=0.7,
           edgecolors='darkred', linewidths=0.4, zorder=5)
ax4.contour(X_ext, Y_ext, predictions_ext, levels=[0], colors='red', linewidths=1, zorder=6)
# Training box
from matplotlib.patches import Rectangle
training_box4 = Rectangle((0, -1.5), 2*np.pi, 3.0, linewidth=2, edgecolor='lime',
                          facecolor='none', linestyle='-', zorder=7)
ax4.add_patch(training_box4)
ax4.axvspan(0, 2*np.pi, alpha=0.12, color='green', zorder=1)
ax4.set_title('Model Prediction\n(Generalization Range)', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(x_min_ext, x_max_ext)
ax4.set_ylim(y_min_ext, y_max_ext)
x_ticks_ext = np.arange(-4*np.pi, 6.5*np.pi, np.pi)
x_labels_ext = [f'{int(tick/np.pi)}π' if tick != 0 else '0' for tick in x_ticks_ext]
ax4.set_xticks(x_ticks_ext)
ax4.set_xticklabels(x_labels_ext, fontsize=9)

# BOTTOM MIDDLE: Actual (Extended)
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.contourf(X_ext, Y_ext, actual_ext, levels=[-1.5, 0, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.6)
x_sine_ext = np.linspace(x_min_ext, x_max_ext, 3000)
y_sine_ext = np.sin(x_sine_ext)
ax5.plot(x_sine_ext, y_sine_ext, 'black', linewidth=2, label='y = sin(x)')
ax5.fill_between(x_sine_ext, y_sine_ext, y_max_ext, alpha=0.3, color='blue')
ax5.fill_between(x_sine_ext, y_sine_ext, y_min_ext, alpha=0.3, color='red')
# Training box
training_box5 = Rectangle((0, -1.5), 2*np.pi, 3.0, linewidth=2, edgecolor='lime',
                          facecolor='none', linestyle='-', zorder=4)
ax5.add_patch(training_box5)
ax5.axvspan(0, 2*np.pi, alpha=0.12, color='green', zorder=1)
ax5.set_title('Actual Sine Wave\n(Generalized Range)', fontsize=14)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(x_min_ext, x_max_ext)
ax5.set_ylim(y_min_ext, y_max_ext)
ax5.set_xticks(x_ticks_ext)
ax5.set_xticklabels(x_labels_ext, fontsize=9)

# BOTTOM RIGHT: Errors (Extended)
ax6 = fig.add_subplot(gs[1, 2])
difference_ext = (predictions_ext != actual_ext).astype(float)
im6 = ax6.contourf(X_ext, Y_ext, difference_ext, levels=[0, 0.5, 1], 
                   colors=['#90EE90', '#FF6B6B'], alpha=0.7)
ax6.contour(X_ext, Y_ext, predictions_ext, levels=[0], colors='red', linewidths=1,
           linestyles='--', alpha=0.9)
ax6.plot(x_sine_ext, y_sine_ext, 'blue', linewidth=1, linestyle='--', alpha=0.9)
# Training box - BRIGHT
training_box6 = Rectangle((0, -1.5), 2*np.pi, 3.0, linewidth=2, edgecolor='lime',
                          facecolor='lime', linestyle='-', alpha=0.25, zorder=3)
ax6.add_patch(training_box6)
# Region labels
ax6.text(-2*np.pi, -3.5, 'FAR LEFT', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
ax6.text(np.pi, -3.5, 'TRAINED', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
ax6.text(4*np.pi, -3.5, 'FAR RIGHT', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
ax6.set_title('Errors\n(Generalization Test)', fontsize=14)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(x_min_ext, x_max_ext)
ax6.set_ylim(y_min_ext, y_max_ext)
ax6.set_xticks(x_ticks_ext)
ax6.set_xticklabels(x_labels_ext, fontsize=9)
legend_elements_ext = [
    Patch(facecolor='#90EE90', label='Correct'),
    Patch(facecolor='#FF6B6B', label='Error')
]
ax6.legend(handles=legend_elements_ext, fontsize=9, loc='upper right')

plt.suptitle('Sine Wave Classification - Training vs Generalization\n' +
             'TOP: Training Region [0, 2π] × [-1.5, 1.5] (High Detail) | ' +
             'BOTTOM: Extended Range [-4π, 6π] × [-4, 4] (Generalization Test)', 
             fontsize=16, fontweight='bold')

# Save
output_path = os.path.join(os.path.dirname(__file__), 'sine_visualization.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()

# Calculate accuracies
train_acc = np.mean(predictions_train == actual_train) * 100
ext_acc = np.mean(predictions_ext == actual_ext) * 100

# Training region in extended view
train_mask_ext = (X_ext >= 0) & (X_ext <= 2*np.pi) & (Y_ext >= -1.5) & (Y_ext <= 1.5)
train_in_ext_acc = np.mean(predictions_ext[train_mask_ext] == actual_ext[train_mask_ext]) * 100

# Generalization only (outside training box)
gen_only_mask = ~train_mask_ext
gen_only_acc = np.mean(predictions_ext[gen_only_mask] == actual_ext[gen_only_mask]) * 100

print(f"\n{'='*80}")
print(f"TRAINING vs GENERALIZATION COMPARISON:")
print(f"{'='*80}")
print(f"TOP ROW - Training Region Detail:")
print(f"  Accuracy in [0, 2π] × [-1.5, 1.5]:        {train_acc:.2f}%")
print(f"-" * 80)
print(f"BOTTOM ROW - Extended Range:")
print(f"  Training box (same region):               {train_in_ext_acc:.2f}%")
print(f"  Generalization ONLY (outside box):        {gen_only_acc:.2f}%")
print(f"  Overall Extended Range:                   {ext_acc:.2f}%")
print(f"{'='*80}")

if gen_only_acc > 93:
    print("✅ EXCELLENT! Model generalizes brilliantly!")
elif gen_only_acc > 87:
    print("⚠️  Model partially generalizes")
else:
    print("❌ Model struggles outside training region")

print(f"\n🔍 Compare TOP row (detailed) vs BOTTOM row (generalization)")
print(f"   Green box in bottom row = training region from top row")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
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

print(f"✅ Loaded {len(train_inputs)} training samples")
print(f"   X range: [{train_inputs[:, 0].min():.2f}, {train_inputs[:, 0].max():.2f}]")
print(f"   Y range: [{train_inputs[:, 1].min():.2f}, {train_inputs[:, 1].max():.2f}]")

# Extended range for bottom row
x_min_ext, x_max_ext = -6*np.pi, 8*np.pi
y_min_ext, y_max_ext = -2.5, 2.5

# Training region for top row
x_min_train, x_max_train = -2*np.pi, 4*np.pi
y_min_train, y_max_train = -1.5, 1.5

print("\nGenerating predictions for BOTH ranges...")

# Create custom colormap: darker red (below) -> white -> darker blue (above)
colors_list = ['#ff9999', '#ffffff', '#9999ff']  # Darker red -> white -> darker blue
n_bins = 256
cmap_name = 'sine_custom'
sine_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=n_bins)

# Create custom green-to-red colormap for errors
error_colors = ['#90EE90', '#FF6B6B']  # Green to Red
error_cmap = LinearSegmentedColormap.from_list('error_custom', error_colors, N=256)

# Generate predictions for extended range - KEEP RAW CONFIDENCE
resolution_x_ext = 1400
resolution_y_ext = 300

x_ext = np.linspace(x_min_ext, x_max_ext, resolution_x_ext)
y_ext = np.linspace(y_min_ext, y_max_ext, resolution_y_ext)
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

predictions_raw_ext = np.zeros((resolution_y_ext, resolution_x_ext))
predictions_binary_ext = np.zeros((resolution_y_ext, resolution_x_ext))
actual_ext = np.zeros((resolution_y_ext, resolution_x_ext))

print("  Extended range predictions...")
for i in range(resolution_y_ext):
    if i % 50 == 0:
        print(f"    Row {i}/{resolution_y_ext}...")
    for j in range(resolution_x_ext):
        point = np.array([X_ext[i, j], Y_ext[i, j]])
        output = neural_net.forward_propagation(point)
        
        # Raw confidence (0 to 1 for sigmoid)
        predictions_raw_ext[i, j] = output[0]
        
        # Binary for accuracy
        predictions_binary_ext[i, j] = 1 if output[0] > 0.5 else 0
        
        sine_value = np.sin(X_ext[i, j])
        actual_ext[i, j] = 1 if Y_ext[i, j] > sine_value else 0

# Calculate confidence-based error for extended range
confidence_error_ext = np.abs(predictions_raw_ext - actual_ext)

# Generate predictions for training region - KEEP RAW CONFIDENCE
resolution_x_train = 900
resolution_y_train = 250

x_train = np.linspace(x_min_train, x_max_train, resolution_x_train)
y_train = np.linspace(y_min_train, y_max_train, resolution_y_train)
X_train, Y_train = np.meshgrid(x_train, y_train)

predictions_raw_train = np.zeros((resolution_y_train, resolution_x_train))
predictions_binary_train = np.zeros((resolution_y_train, resolution_x_train))
actual_train = np.zeros((resolution_y_train, resolution_x_train))

print("  Training region predictions (3 periods, high detail)...")
for i in range(resolution_y_train):
    if i % 50 == 0:
        print(f"    Row {i}/{resolution_y_train}...")
    for j in range(resolution_x_train):
        point = np.array([X_train[i, j], Y_train[i, j]])
        output = neural_net.forward_propagation(point)
        
        # Raw confidence (0 to 1 for sigmoid)
        predictions_raw_train[i, j] = output[0]
        
        # Binary for accuracy
        predictions_binary_train[i, j] = 1 if output[0] > 0.5 else 0
        
        sine_value = np.sin(X_train[i, j])
        actual_train[i, j] = 1 if Y_train[i, j] > sine_value else 0

# Calculate confidence-based error for training region
confidence_error_train = np.abs(predictions_raw_train - actual_train)

print("Rendering visualization...")

# Create figure
fig = plt.figure(figsize=(30, 17))
gs = fig.add_gridspec(2, 3, hspace=0.175, wspace=0.15)

# ============================================
# TOP ROW: TRAINING REGION (3 Periods)
# ============================================

# TOP LEFT: Model Prediction (Training Region) - RAW CONFIDENCE
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(predictions_raw_train, extent=[x_min_train, x_max_train, y_min_train, y_max_train], 
                 origin='lower', cmap=sine_cmap, vmin=0, vmax=1, aspect='auto')
ax1.scatter(above_samples[:, 0], above_samples[:, 1], c='blue', s=8, alpha=0.6, 
           edgecolors='darkblue', linewidths=0.5, zorder=5)
ax1.scatter(below_samples[:, 0], below_samples[:, 1], c='red', s=8, alpha=0.6,
           edgecolors='darkred', linewidths=0.5, zorder=5)
ax1.contour(X_train, Y_train, predictions_raw_train, levels=[0.5], colors='black', 
           linewidths=1.5, alpha=0.4, zorder=10)
ax1.set_xlabel('X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
ax1.set_title('Model Prediction\n(Training Region: 3 Periods)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min_train, x_max_train)
ax1.set_ylim(y_min_train, y_max_train)
x_ticks_train = np.arange(-2*np.pi, 4*np.pi + 0.01, np.pi/2)
x_labels_train = []
for tick in x_ticks_train:
    val = tick / np.pi
    if val == int(val):
        x_labels_train.append(f'{int(val)}π')
    else:
        x_labels_train.append(f'{val:.1f}π')
x_labels_train[x_labels_train.index('0π')] = '0'
ax1.set_xticks(x_ticks_train)
ax1.set_xticklabels(x_labels_train, fontsize=9)

# Add colorbar for plot 1
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# TOP MIDDLE: Actual (Training Region)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(actual_train, extent=[x_min_train, x_max_train, y_min_train, y_max_train], 
                 origin='lower', cmap=sine_cmap, vmin=0, vmax=1, aspect='auto')
x_sine_train = np.linspace(x_min_train, x_max_train, 1000)
y_sine_train = np.sin(x_sine_train)
ax2.plot(x_sine_train, y_sine_train, 'black', linewidth=2, label='y = sin(x)', linestyle='--')
ax2.set_xlabel('X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y', fontsize=12, fontweight='bold')
ax2.set_title('Actual Sine Wave\n(Training Region: 3 Periods)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(x_min_train, x_max_train)
ax2.set_ylim(y_min_train, y_max_train)
ax2.set_xticks(x_ticks_train)
ax2.set_xticklabels(x_labels_train, fontsize=9)

# Add invisible colorbar to maintain consistent sizing
dummy_cbar2 = plt.colorbar(cm.ScalarMappable(cmap=sine_cmap), ax=ax2, fraction=0.046, pad=0.04)
dummy_cbar2.ax.set_visible(False)

# TOP RIGHT: Errors (Training Region)
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(confidence_error_train, extent=[x_min_train, x_max_train, y_min_train, y_max_train], 
                 origin='lower', cmap=error_cmap, vmin=0, vmax=1, aspect='auto')
ax3.contour(X_train, Y_train, predictions_raw_train, levels=[0.5], colors='red', 
           linewidths=2, linestyles='--', alpha=0.9)
ax3.plot(x_sine_train, y_sine_train, 'blue', linewidth=2, linestyle='--', alpha=0.9)
ax3.set_xlabel('X', fontsize=12, fontweight='bold')
ax3.set_ylabel('Y', fontsize=12, fontweight='bold')
ax3.set_title('Errors (Model vs Actual)\n(Training Region)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(x_min_train, x_max_train)
ax3.set_ylim(y_min_train, y_max_train)
ax3.set_xticks(x_ticks_train)
ax3.set_xticklabels(x_labels_train, fontsize=9)
from matplotlib.patches import Patch
legend_elements_train = [
    Patch(facecolor='#90EE90', label='Low Error'),
    Patch(facecolor='#FF6B6B', label='High Error'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Model'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='True'),
]
ax3.legend(handles=legend_elements_train, fontsize=10, loc='upper left')

# Add colorbar for plot 3
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Absolute Error', fontsize=11)

# ============================================
# BOTTOM ROW: EXTENDED RANGE (7 Periods Total)
# ============================================

# BOTTOM LEFT: Model Prediction (Extended) - RAW CONFIDENCE
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(predictions_raw_ext, extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext], 
                 origin='lower', cmap=sine_cmap, vmin=0, vmax=1, aspect='auto')
ax4.scatter(above_samples[:, 0], above_samples[:, 1], c='blue', s=5, alpha=0.7,
           edgecolors='darkblue', linewidths=0.3, zorder=5)
ax4.scatter(below_samples[:, 0], below_samples[:, 1], c='red', s=5, alpha=0.7,
           edgecolors='darkred', linewidths=0.3, zorder=5)
ax4.contour(X_ext, Y_ext, predictions_raw_ext, levels=[0.5], colors='black', 
           linewidths=1.5, alpha=0.4, zorder=10)

# Training box overlay
from matplotlib.patches import Rectangle
training_box4 = Rectangle((-2*np.pi, -1.5), 6*np.pi, 3.0, linewidth=3, edgecolor='lime',
                          facecolor='green', alpha=0.12, linestyle='-', zorder=1) 
ax4.add_patch(training_box4)

# Border only
training_box4_border = Rectangle((-2*np.pi, -1.5), 6*np.pi, 3.0, linewidth=3, edgecolor='lime',
                                 facecolor='none', linestyle='-', zorder=4)
ax4.add_patch(training_box4_border)

ax4.set_xlabel('X', fontsize=12, fontweight='bold')
ax4.set_ylabel('Y', fontsize=12, fontweight='bold')
ax4.set_title('Model Prediction\n(Generalization: 7 Periods)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(x_min_ext, x_max_ext)
ax4.set_ylim(y_min_ext, y_max_ext)
x_ticks_ext = np.arange(-6*np.pi, 8.5*np.pi, np.pi)
x_labels_ext = [f'{int(tick/np.pi)}π' if tick != 0 else '0' for tick in x_ticks_ext]
ax4.set_xticks(x_ticks_ext)
ax4.set_xticklabels(x_labels_ext, fontsize=9)

# Add colorbar for plot 4
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# BOTTOM MIDDLE: Actual (Extended)
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(actual_ext, extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext], 
                 origin='lower', cmap=sine_cmap, vmin=0, vmax=1, aspect='auto')
x_sine_ext = np.linspace(x_min_ext, x_max_ext, 3000)
y_sine_ext = np.sin(x_sine_ext)
ax5.plot(x_sine_ext, y_sine_ext, 'black', linewidth=2, label='y = sin(x)', linestyle='--')

# Training box overlay
training_box5 = Rectangle((-2*np.pi, -1.5), 6*np.pi, 3.0, linewidth=3, edgecolor='lime',
                          facecolor='green', alpha=0.12, linestyle='-', zorder=1) 
ax5.add_patch(training_box5)

# Border only
training_box5_border = Rectangle((-2*np.pi, -1.5), 6*np.pi, 3.0, linewidth=3, edgecolor='lime',
                                 facecolor='none', linestyle='-', zorder=4)
ax5.add_patch(training_box5_border)

ax5.set_xlabel('X', fontsize=12, fontweight='bold')
ax5.set_ylabel('Y', fontsize=12, fontweight='bold')
ax5.set_title('Actual Sine Wave\n(Generalization: 7 Periods)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10, loc='upper left') 
ax5.grid(True, alpha=0.3)
ax5.set_xlim(x_min_ext, x_max_ext)
ax5.set_ylim(y_min_ext, y_max_ext)
ax5.set_xticks(x_ticks_ext)
ax5.set_xticklabels(x_labels_ext, fontsize=9)

# Add invisible colorbar to maintain consistent sizing
dummy_cbar5 = plt.colorbar(cm.ScalarMappable(cmap=sine_cmap), ax=ax5, fraction=0.046, pad=0.04)
dummy_cbar5.ax.set_visible(False)

# BOTTOM RIGHT: Errors (Extended)
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.imshow(confidence_error_ext, extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext], 
                 origin='lower', cmap=error_cmap, vmin=0, vmax=1, aspect='auto')
ax6.contour(X_ext, Y_ext, predictions_raw_ext, levels=[0.5], colors='red', linewidths=2,
           linestyles='--', alpha=0.9)
ax6.plot(x_sine_ext, y_sine_ext, 'blue', linewidth=2, linestyle='--', alpha=0.9)

# Training box
training_box6 = Rectangle((-2*np.pi, -1.5), 6*np.pi, 3.0, linewidth=3, edgecolor='lime',
                          facecolor='lime', linestyle='-', alpha=0.25, zorder=3)
ax6.add_patch(training_box6)

# Region labels - MOVED TO BOTTOM
ax6.text(-4*np.pi, -2.2, 'FAR LEFT\n(2 periods)', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
ax6.text(np.pi, -2.2, 'TRAINED\n(3 periods)', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
ax6.text(6*np.pi, -2.2, 'FAR RIGHT\n(2 periods)', fontsize=11, ha='center', fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

ax6.set_xlabel('X', fontsize=12, fontweight='bold')
ax6.set_ylabel('Y', fontsize=12, fontweight='bold')
ax6.set_title('Errors (Model vs Actual)\n(Generalized Region)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(x_min_ext, x_max_ext)
ax6.set_ylim(y_min_ext, y_max_ext)
ax6.set_xticks(x_ticks_ext)
ax6.set_xticklabels(x_labels_ext, fontsize=9)
legend_elements_ext = [
    Patch(facecolor='#90EE90', label='Low Error'),
    Patch(facecolor='#FF6B6B', label='High Error'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Model'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='True'),
]
ax6.legend(handles=legend_elements_ext, fontsize=9, loc='upper left')

# Add colorbar for plot 6
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.set_label('Absolute Error', fontsize=11)

plt.suptitle('Sine Wave Classification - Training (3 Periods) vs Generalization (7 Periods)\n' +
             'TOP: Training Region [-2π, 4π] × [-1.5, 1.5] | ' +
             'BOTTOM: Extended Range [-6π, 8π] × [-2.5, 2.5]', 
             fontsize=16, fontweight='bold', y=0.94)

# Adjust layout to make room for title
plt.tight_layout(rect=[0, 0, 1, 0.95]) 

# Save
output_path = os.path.join(os.path.dirname(__file__), 'sine_visualization.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()

# Calculate accuracies
train_acc = np.mean(predictions_binary_train == actual_train) * 100
ext_acc = np.mean(predictions_binary_ext == actual_ext) * 100

train_mask_ext = (X_ext >= -2*np.pi) & (X_ext <= 4*np.pi) & (Y_ext >= -1.5) & (Y_ext <= 1.5)
train_in_ext_acc = np.mean(predictions_binary_ext[train_mask_ext] == actual_ext[train_mask_ext]) * 100

gen_only_mask = ~train_mask_ext
gen_only_acc = np.mean(predictions_binary_ext[gen_only_mask] == actual_ext[gen_only_mask]) * 100

# Calculate mean errors
mean_error_train = np.mean(confidence_error_train)
mean_error_ext = np.mean(confidence_error_ext)
mean_error_gen_only = np.mean(confidence_error_ext[gen_only_mask])

print(f"\n{'='*80}")
print(f"TRAINING vs GENERALIZATION COMPARISON:")
print(f"{'='*80}")
print(f"TOP ROW - Training Region (3 periods):")
print(f"  Accuracy in [-2π, 4π] × [-1.5, 1.5]:      {train_acc:.2f}%")
print(f"  Mean Prediction Error:                    {mean_error_train:.3f}")
print(f"-" * 80)
print(f"BOTTOM ROW - Extended Range (7 periods):")
print(f"  Training box (same region):               {train_in_ext_acc:.2f}%")
print(f"  Generalization ONLY (outside box):        {gen_only_acc:.2f}%")
print(f"  Mean Error (gen only):                    {mean_error_gen_only:.3f}")
print(f"  Overall Extended Range:                   {ext_acc:.2f}%")
print(f"  Mean Prediction Error:                    {mean_error_ext:.3f}")
print(f"{'='*80}")

if gen_only_acc > 93:
    print("✅ EXCELLENT! Model generalizes brilliantly to unseen periods!")
elif gen_only_acc > 87:
    print("⚠️  Model partially generalizes")
else:
    print("❌ Model struggles outside training region")

print(f"\n🔍 Compare TOP row (3 trained periods) vs BOTTOM row (7 total periods)")
print(f"   Green box = training region [-2π, 4π]")
print(f"   Orange regions = pure generalization (never seen during training)")
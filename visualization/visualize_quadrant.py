import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_quadrant.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quadrant_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['Input_Values'])
train_outputs = np.array(data['Output_Values'])

# Convert one-hot encoding to class labels (0-3)
train_classes = np.argmax(train_outputs, axis=1)

print(f"✅ Loaded {len(train_inputs)} training samples")
print(f"   X range: [{train_inputs[:, 0].min():.2f}, {train_inputs[:, 0].max():.2f}]")
print(f"   Y range: [{train_inputs[:, 1].min():.2f}, {train_inputs[:, 1].max():.2f}]")

# Define colors for each quadrant
colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']  # Red, Green, Blue, Yellow
quadrant_names = ['Q1 (x>0, y>0)', 'Q2 (x<0, y>0)', 'Q3 (x<0, y<0)', 'Q4 (x>0, y<0)']

# Training region
x_min_train, x_max_train = -5, 5
y_min_train, y_max_train = -5, 5

# Extended region (2.5x larger)
x_min_ext, x_max_ext = -12.5, 12.5
y_min_ext, y_max_ext = -12.5, 12.5

print("\nGenerating predictions for BOTH ranges...")

# ============================================
# Generate predictions for TRAINING region
# ============================================
resolution_train = 700
x_train = np.linspace(x_min_train, x_max_train, resolution_train)
y_train = np.linspace(y_min_train, y_max_train, resolution_train)
X_train, Y_train = np.meshgrid(x_train, y_train)

predictions_train = np.zeros((resolution_train, resolution_train))
confidences_train = np.zeros((resolution_train, resolution_train))
all_outputs_train = np.zeros((resolution_train, resolution_train, 4))

print("  Training region predictions...")
for i in range(resolution_train):
    if i % 100 == 0:
        print(f"    Row {i}/{resolution_train}...")
    for j in range(resolution_train):
        point = np.array([X_train[i, j], Y_train[i, j]])
        output = neural_net.forward_propagation(point)
        
        predictions_train[i, j] = np.argmax(output)
        confidences_train[i, j] = np.max(output)
        all_outputs_train[i, j] = output

# Create actual quadrant pattern for training region
actual_train = np.zeros((resolution_train, resolution_train))
for i in range(resolution_train):
    for j in range(resolution_train):
        x_val, y_val = X_train[i, j], Y_train[i, j]
        if x_val > 0 and y_val > 0:
            actual_train[i, j] = 0  # Q1
        elif x_val < 0 and y_val > 0:
            actual_train[i, j] = 1  # Q2
        elif x_val < 0 and y_val < 0:
            actual_train[i, j] = 2  # Q3
        else:
            actual_train[i, j] = 3  # Q4

# Calculate confidence on CORRECT answer for training region
correct_class_confidence_train = np.zeros((resolution_train, resolution_train))
for i in range(resolution_train):
    for j in range(resolution_train):
        correct_class = int(actual_train[i, j])
        correct_class_confidence_train[i, j] = all_outputs_train[i, j, correct_class]

# ============================================
# Generate predictions for EXTENDED region
# ============================================
resolution_ext = 700
x_ext = np.linspace(x_min_ext, x_max_ext, resolution_ext)
y_ext = np.linspace(y_min_ext, y_max_ext, resolution_ext)
X_ext, Y_ext = np.meshgrid(x_ext, y_ext)

predictions_ext = np.zeros((resolution_ext, resolution_ext))
confidences_ext = np.zeros((resolution_ext, resolution_ext))
all_outputs_ext = np.zeros((resolution_ext, resolution_ext, 4))

print("  Extended region predictions...")
for i in range(resolution_ext):
    if i % 100 == 0:
        print(f"    Row {i}/{resolution_ext}...")
    for j in range(resolution_ext):
        point = np.array([X_ext[i, j], Y_ext[i, j]])
        output = neural_net.forward_propagation(point)
        
        predictions_ext[i, j] = np.argmax(output)
        confidences_ext[i, j] = np.max(output)
        all_outputs_ext[i, j] = output

# Create actual quadrant pattern for extended region
actual_ext = np.zeros((resolution_ext, resolution_ext))
for i in range(resolution_ext):
    for j in range(resolution_ext):
        x_val, y_val = X_ext[i, j], Y_ext[i, j]
        if x_val > 0 and y_val > 0:
            actual_ext[i, j] = 0  # Q1
        elif x_val < 0 and y_val > 0:
            actual_ext[i, j] = 1  # Q2
        elif x_val < 0 and y_val < 0:
            actual_ext[i, j] = 2  # Q3
        else:
            actual_ext[i, j] = 3  # Q4

# Calculate confidence on CORRECT answer for extended region
correct_class_confidence_ext = np.zeros((resolution_ext, resolution_ext))
for i in range(resolution_ext):
    for j in range(resolution_ext):
        correct_class = int(actual_ext[i, j])
        correct_class_confidence_ext[i, j] = all_outputs_ext[i, j, correct_class]

print("Rendering visualization...")

# Create figure with 2 rows, 3 columns
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.17)

# ============================================
# TOP ROW: TRAINING REGION
# ============================================

# TOP LEFT: Model's Prediction with Confidence
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(X_train, Y_train, predictions_train, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Overlay confidence as contours (darker = more confident)
conf_contours = ax1.imshow(confidences_train, extent=[x_min_train, x_max_train, y_min_train, y_max_train], 
                           origin='lower', cmap='Greys', alpha=0.6, interpolation='bilinear')

# Overlay training points
for q in range(4):
    mask = train_classes == q
    ax1.scatter(train_inputs[mask, 0], train_inputs[mask, 1],
               c=colors[q], label=quadrant_names[q], s=5, alpha=0.5, 
               edgecolors='white', linewidths=0.3)

ax1.set_xlabel('X', fontsize=14)
ax1.set_ylabel('Y', fontsize=14)
ax1.set_title('Model Prediction\n(Training Region, Darker = More Confident)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right', markerscale=3)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(x_min_train, x_max_train)
ax1.set_ylim(y_min_train, y_max_train)

# Add colorbar for confidence
cbar1 = plt.colorbar(conf_contours, ax=ax1, fraction=0.046, pad=0.04)

# TOP MIDDLE: Actual Quadrants
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(X_train, Y_train, actual_train, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Add quadrant labels
ax2.text(2.5, 2.5, 'Q1\n(x>0, y>0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(-2.5, 2.5, 'Q2\n(x<0, y>0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(-2.5, -2.5, 'Q3\n(x<0, y<0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(2.5, -2.5, 'Q4\n(x>0, y<0)', fontsize=14, ha='center', va='center', fontweight='bold')

ax2.set_xlabel('X', fontsize=14)
ax2.set_ylabel('Y', fontsize=14)
ax2.set_title('Actual Quadrants\n(Training Region)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(x_min_train, x_max_train)
ax2.set_ylim(y_min_train, y_max_train)

# Add invisible colorbar to maintain consistent sizing
dummy_cbar2 = plt.colorbar(cm.ScalarMappable(cmap='Greys'), ax=ax2, fraction=0.046, pad=0.04)
dummy_cbar2.ax.set_visible(False)

# TOP RIGHT: Confidence on Correct Answer
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(correct_class_confidence_train, extent=[x_min_train, x_max_train, y_min_train, y_max_train], 
                 origin='lower', cmap='RdYlGn', vmin=0, vmax=1, interpolation='bilinear')

ax3.set_xlabel('X', fontsize=14)
ax3.set_ylabel('Y', fontsize=14)
ax3.set_title('Confidence in Correct Quadrant\n(Training Region)', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.set_xlim(x_min_train, x_max_train)
ax3.set_ylim(y_min_train, y_max_train)

# Add legend for color meanings
from matplotlib.patches import Patch
legend_elements_train = [
    Patch(facecolor='green', label='High Confidence/Correct'),
    Patch(facecolor='red', label='Low Confidence/Wrong'),
]
ax3.legend(handles=legend_elements_train, fontsize=10, loc='upper left')

# Add colorbar
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Probability of Correct Class', rotation=270, labelpad=15, fontsize=10)

# ============================================
# BOTTOM ROW: EXTENDED REGION (2.5x larger)
# ============================================

# BOTTOM LEFT: Model's Prediction with Confidence (Extended)
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.contourf(X_ext, Y_ext, predictions_ext, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Overlay confidence as contours
conf_contours_ext = ax4.imshow(confidences_ext, extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext], 
                               origin='lower', cmap='Greys', alpha=0.6, interpolation='bilinear')

# Overlay training points
for q in range(4):
    mask = train_classes == q
    ax4.scatter(train_inputs[mask, 0], train_inputs[mask, 1],
               c=colors[q], s=3, alpha=0.7, edgecolors='white', linewidths=0.2, zorder=5)

# Training box overlay
training_box4 = Rectangle((x_min_train, y_min_train), x_max_train - x_min_train, y_max_train - y_min_train, 
                          linewidth=3, edgecolor='lime', facecolor='green', alpha=0.12, linestyle='-', zorder=1)
ax4.add_patch(training_box4)

# Border only
training_box4_border = Rectangle((x_min_train, y_min_train), x_max_train - x_min_train, y_max_train - y_min_train,
                                 linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', zorder=4)
ax4.add_patch(training_box4_border)

ax4.set_xlabel('X', fontsize=14)
ax4.set_ylabel('Y', fontsize=14)
ax4.set_title('Model Prediction\n(Generalization: 2.5x Range)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.2)
ax4.set_xlim(x_min_ext, x_max_ext)
ax4.set_ylim(y_min_ext, y_max_ext)

# Add colorbar for confidence
cbar4 = plt.colorbar(conf_contours_ext, ax=ax4, fraction=0.046, pad=0.04)

# BOTTOM MIDDLE: Actual Quadrants (Extended)
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.contourf(X_ext, Y_ext, actual_ext, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Training box overlay
training_box5 = Rectangle((x_min_train, y_min_train), x_max_train - x_min_train, y_max_train - y_min_train,
                          linewidth=3, edgecolor='lime', facecolor='green', alpha=0.12, linestyle='-', zorder=1)
ax5.add_patch(training_box5)

# Border only
training_box5_border = Rectangle((x_min_train, y_min_train), x_max_train - x_min_train, y_max_train - y_min_train,
                                 linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', zorder=4)
ax5.add_patch(training_box5_border)

ax5.set_xlabel('X', fontsize=14)
ax5.set_ylabel('Y', fontsize=14)
ax5.set_title('Actual Quadrants\n(Generalization: 2.5x Range)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.2)
ax5.set_xlim(x_min_ext, x_max_ext)
ax5.set_ylim(y_min_ext, y_max_ext)

# Add invisible colorbar to maintain consistent sizing
dummy_cbar5 = plt.colorbar(cm.ScalarMappable(cmap='Greys'), ax=ax5, fraction=0.046, pad=0.04)
dummy_cbar5.ax.set_visible(False)

# BOTTOM RIGHT: Confidence on Correct Answer (Extended)
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.imshow(correct_class_confidence_ext, extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext],
                 origin='lower', cmap='RdYlGn', vmin=0, vmax=1, interpolation='bilinear')

# Training box
training_box6 = Rectangle((x_min_train, y_min_train), x_max_train - x_min_train, y_max_train - y_min_train,
                          linewidth=3, edgecolor='lime', facecolor='lime', linestyle='-', alpha=0.25, zorder=3)
ax6.add_patch(training_box6)

# Region labels
ax6.text(0, -10, 'TRAINING REGION\n(within green box)', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
ax6.text(6.5, 9, 'GENERALIZATION\n(outside box)', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.9))

ax6.set_xlabel('X', fontsize=14)
ax6.set_ylabel('Y', fontsize=14)
ax6.set_title('Confidence in Correct Quadrant\n(Generalization Region)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.2)
ax6.set_xlim(x_min_ext, x_max_ext)
ax6.set_ylim(y_min_ext, y_max_ext)

# Add legend for color meanings
legend_elements_ext = [
    Patch(facecolor='green', label='High Confidence/Correct'),
    Patch(facecolor='red', label='Low Confidence/Wrong'),
]
ax6.legend(handles=legend_elements_ext, fontsize=10, loc='upper left')

# Add colorbar
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.set_label('Probability of Correct Class', rotation=270, labelpad=15, fontsize=10)

plt.suptitle('Quadrant Classification - Training vs Generalization\n' +
             'TOP: Training Region [-5, 5] × [-5, 5] | BOTTOM: Extended Range [-12.5, 12.5] × [-12.5, 12.5]', 
             fontsize=16, fontweight='bold', y=0.97)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save
output_path = os.path.join(os.path.dirname(__file__), 'quadrant_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Calculate accuracies
train_acc = np.mean(predictions_train == actual_train) * 100
ext_acc = np.mean(predictions_ext == actual_ext) * 100

# Calculate accuracy within training box in extended region
train_mask_ext = (X_ext >= x_min_train) & (X_ext <= x_max_train) & (Y_ext >= y_min_train) & (Y_ext <= y_max_train)
train_in_ext_acc = np.mean(predictions_ext[train_mask_ext] == actual_ext[train_mask_ext]) * 100

# Calculate accuracy outside training box (generalization only)
gen_only_mask = ~train_mask_ext
gen_only_acc = np.mean(predictions_ext[gen_only_mask] == actual_ext[gen_only_mask]) * 100

# Mean confidences
mean_confidence_train = np.mean(confidences_train)
mean_correct_confidence_train = np.mean(correct_class_confidence_train)
mean_confidence_ext = np.mean(confidences_ext)
mean_correct_confidence_ext = np.mean(correct_class_confidence_ext)
mean_correct_confidence_gen_only = np.mean(correct_class_confidence_ext[gen_only_mask])

print(f"\n{'='*80}")
print(f"TRAINING vs GENERALIZATION COMPARISON:")
print(f"{'='*80}")
print(f"TOP ROW - Training Region [-5, 5]:")
print(f"  Accuracy:                                 {train_acc:.2f}%")
print(f"  Mean Confidence (all predictions):        {mean_confidence_train:.3f}")
print(f"  Mean Confidence on CORRECT answer:        {mean_correct_confidence_train:.3f}")
print(f"-" * 80)
print(f"BOTTOM ROW - Extended Range [-12.5, 12.5] (2.5x larger):")
print(f"  Training box (same region):               {train_in_ext_acc:.2f}%")
print(f"  Generalization ONLY (outside box):        {gen_only_acc:.2f}%")
print(f"  Mean Conf on Correct (gen only):          {mean_correct_confidence_gen_only:.3f}")
print(f"  Overall Extended Range:                   {ext_acc:.2f}%")
print(f"  Mean Confidence (all predictions):        {mean_confidence_ext:.3f}")
print(f"  Mean Confidence on CORRECT answer:        {mean_correct_confidence_ext:.3f}")
print(f"{'='*80}")

if gen_only_acc > 95:
    print("✅ EXCELLENT! Model generalizes perfectly to unseen regions!")
elif gen_only_acc > 85:
    print("⚠️  Model partially generalizes")
else:
    print("❌ Model struggles outside training region")

print(f"\n🔍 Compare TOP row (training) vs BOTTOM row (2.5x extended)")
print(f"   Green box = training region [-5, 5]")
print(f"   Orange region = pure generalization (never seen during training)")

# Per-quadrant accuracy in extended region
print(f"\n{'='*80}")
print(f"PER-QUADRANT ACCURACY (Extended Region):")
print(f"{'='*80}")
for q in range(4):
    mask_actual = actual_ext == q
    correct = np.sum((predictions_ext == q) & (actual_ext == q))
    total = np.sum(mask_actual)
    acc = (correct / total) * 100 if total > 0 else 0
    avg_conf = np.mean(correct_class_confidence_ext[mask_actual])
    print(f"{quadrant_names[q]}: {acc:.2f}% accuracy, avg confidence on correct = {avg_conf:.3f}")
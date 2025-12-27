import numpy as np
import plotly.graph_objects as go
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load model and data
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_iris.json')
neural_net.load(model_path)

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['Input_Values'])
train_outputs = np.array(data['Output_Values'])
train_classes = np.argmax(train_outputs, axis=1)

class_names = ['Setosa', 'Versicolor', 'Virginica']
colors_actual = ['red', 'green', 'blue']

# 3D features to display: Sepal Length (0), Petal Length (2), Petal Width (3)
# 4th feature for slider: Sepal Width (1)
display_features = [0, 2, 3]
slider_feature = 1
feature_names_all = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
feature_names_3d = [feature_names_all[i] for i in display_features]
slider_feature_name = feature_names_all[slider_feature]

print(f"Creating 4D visualization:")
print(f"  3D axes: {feature_names_3d}")
print(f"  Slider: {slider_feature_name}")

# Check the range of slider feature in training data
slider_min = train_inputs[:, slider_feature].min()
slider_max = train_inputs[:, slider_feature].max()
print(f"  Training data {slider_feature_name} range: [{slider_min:.3f}, {slider_max:.3f}]")

# Generate frames for different slider values (25 positions!)
slider_values = np.linspace(0, 1, 25)
frames = []

resolution = 25
sample_size_per_frame = 3000
tolerance = 0.05  # ±0.05 around slider value

for slider_idx, slider_val in enumerate(slider_values):
    # Sample points in 3D space for decision regions
    decision_points = {0: [], 1: [], 2: []}
    
    for _ in range(sample_size_per_frame):
        # Random point in 3D
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        z = np.random.uniform(0, 1)
        
        # Create full 4D input with slider feature fixed
        point = np.zeros(4)
        point[display_features[0]] = x
        point[display_features[1]] = y
        point[display_features[2]] = z
        point[slider_feature] = slider_val  # Fixed by slider
        
        # Predict
        output = neural_net.forward_propagation(point)
        predicted_class = np.argmax(output)
        
        decision_points[predicted_class].append([x, y, z])
    
    # Convert to arrays
    for class_idx in range(3):
        decision_points[class_idx] = np.array(decision_points[class_idx])
    
    # Count training points in this slice
    total_points_in_slice = 0
    for class_idx in range(3):
        mask = train_classes == class_idx
        slider_mask = np.abs(train_inputs[:, slider_feature] - slider_val) < tolerance
        combined_mask = mask & slider_mask
        total_points_in_slice += np.sum(combined_mask)
    
    print(f"  Frame {slider_idx + 1}/25: {slider_feature_name} = {slider_val:.2f}, {total_points_in_slice} training points in slice")
    
    # Create traces for this frame
    frame_data = []
    
    # Decision regions (HALF transparent - 0.75 opacity)
    region_colors = ['rgba(255,100,100,0.75)', 'rgba(100,255,100,0.75)', 'rgba(100,150,255,0.75)']
    for class_idx in range(3):
        if len(decision_points[class_idx]) > 0:
            frame_data.append(go.Scatter3d(
                x=decision_points[class_idx][:, 0],
                y=decision_points[class_idx][:, 1],
                z=decision_points[class_idx][:, 2],
                mode='markers',
                name=f'{class_names[class_idx]} region',
                marker=dict(
                    size=3,
                    color=region_colors[class_idx],
                ),
                showlegend=bool(slider_idx == 0)
            ))
    
    # Training data points - ONLY show points NEAR current slider value
    for class_idx in range(3):
        mask = train_classes == class_idx
        # Filter by slider feature value
        slider_mask = np.abs(train_inputs[:, slider_feature] - slider_val) < tolerance
        combined_mask = mask & slider_mask
        
        # ALWAYS add trace, even if empty (for consistent frame structure)
        if np.sum(combined_mask) > 0:
            frame_data.append(go.Scatter3d(
                x=train_inputs[combined_mask, display_features[0]],
                y=train_inputs[combined_mask, display_features[1]],
                z=train_inputs[combined_mask, display_features[2]],
                mode='markers',
                name=f'{class_names[class_idx]} (data)',
                marker=dict(
                    size=5, 
                    color=colors_actual[class_idx],
                    opacity=1.0,
                    line=dict(color='black', width=2)
                ),
                showlegend=bool(slider_idx == 0)
            ))
        else:
            # Add empty trace to keep structure consistent
            frame_data.append(go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode='markers',
                name=f'{class_names[class_idx]} (data)',
                marker=dict(
                    size=10,
                    color=colors_actual[class_idx],
                    opacity=1.0,
                    line=dict(color='black', width=2)
                ),
                showlegend=bool(slider_idx == 0)
            ))
    
    frames.append(go.Frame(data=frame_data, name=f'{slider_val:.3f}'))

print("Assembling interactive visualization...")

# Create initial figure (first frame)
fig = go.Figure(data=frames[0].data, frames=frames)

# Add slider - SHOW ALL LABELS
sliders = [dict(
    active=0,
    yanchor="top",
    y=0.02,
    xanchor="left",
    x=0.05,
    currentvalue=dict(
        prefix=f"{slider_feature_name}: ",
        visible=True,
        xanchor="left",
        font=dict(size=16, color='black')
    ),
    pad=dict(b=10, t=50),
    len=0.9,
    steps=[dict(
        args=[[f.name], dict(
            frame=dict(duration=0, redraw=True),
            mode="immediate",
            transition=dict(duration=0)
        )],
        label=f'{slider_val:.2f}',
        method="animate"
    ) for slider_val, f in zip(slider_values, frames)]
)]

# Update layout
fig.update_layout(
    title=dict(
        text=f'Iris Classification in 4D Feature Space (Interactive Slice)<br>' +
             f'<sub>Slide to vary {slider_feature_name} | Rotate with mouse | Showing data within ±{tolerance:.2f}</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    scene=dict(
        xaxis=dict(title=feature_names_3d[0], range=[0, 1], backgroundcolor='rgb(240, 240, 240)'),
        yaxis=dict(title=feature_names_3d[1], range=[0, 1], backgroundcolor='rgb(240, 240, 240)'),
        zaxis=dict(title=feature_names_3d[2], range=[0, 1], backgroundcolor='rgb(240, 240, 240)'),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        ),
        aspectmode='cube'
    ),
    width=1400,
    height=900,
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='black',
        borderwidth=2,
        font=dict(size=12)
    ),
    sliders=sliders
)

# Save as HTML
output_path = os.path.join(os.path.dirname(__file__), 'iris_visualization_4d.html')
fig.write_html(output_path)
print(f"\n✅ Interactive 4D visualization saved to: {output_path}")
print(f"\n💡 Note: Some slider positions may show few/no training points")
print(f"   This is NORMAL - training data doesn't cover all {slider_feature_name} values equally!")

fig.show()

# Print accuracy
correct = 0
for i in range(len(train_inputs)):
    output = neural_net.forward_propagation(train_inputs[i])
    if np.argmax(output) == train_classes[i]:
        correct += 1

print(f"\n📊 Model Accuracy: {(correct/len(train_inputs)*100):.2f}%")
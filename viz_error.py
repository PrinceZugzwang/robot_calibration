import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")

# Function to plot a coordinate frame
def plot_frame(ax, T, scale=0.1, label=None):
    origin = T[:3, 3]
    x_axis = origin + scale * T[:3, 0]
    y_axis = origin + scale * T[:3, 1]
    z_axis = origin + scale * T[:3, 2]
    
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r-', linewidth=2)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g-', linewidth=2)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b-', linewidth=2)
    
    if label:
        ax.text(origin[0], origin[1], origin[2], label)

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot camera frame
plot_frame(ax, T_cam_to_base, scale=0.2, label="Camera")

# Plot all checkerboard positions
for i, T_cb in enumerate(T_cb_in_base_all):
    if i % 3 == 0:  # Plot every 3rd frame to avoid clutter
        plot_frame(ax, T_cb, scale=0.1, label=f"Board {i}")

# Plot world origin
T_origin = np.eye(4)
plot_frame(ax, T_origin, scale=0.3, label="Base")

# Set equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera and Checkerboard Poses in Base Frame')

# Create a grid
max_range = 1.0
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(0, max_range)

plt.savefig('calibration_visualization.png')
plt.show()
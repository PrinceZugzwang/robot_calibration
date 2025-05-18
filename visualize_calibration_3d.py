import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")   # True checkerboard poses (base frame)
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")   # Checkerboard poses in camera frame
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")    # Calibrated transform


# Compute estimated checkerboard poses in base frame
estimated_positions = []
true_positions = []

for T_cb_base, T_cb_cam in zip(T_cb_in_base_all, T_cb_in_cam_all):
    T_cb_est_base = T_cam_to_base @ T_cb_cam

    p_true = T_cb_base[:3, 3]
    p_est  = T_cb_est_base[:3, 3]

    true_positions.append(p_true)
    estimated_positions.append(p_est)

true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot true checkerboard positions
ax.scatter(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], c='g', label='True (MuJoCo)', s=40)

# Plot estimated positions from calibration
ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], c='r', label='Estimated (T_cam_to_base)', s=40)

# Plot error lines
for true, est in zip(true_positions, estimated_positions):
    ax.plot([true[0], est[0]], [true[1], est[1]], [true[2], est[2]], c='gray', linestyle='--')

# Labels
ax.set_title("Calibration Error: True vs. Estimated Checkerboard Positions")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
plt.tight_layout()
plt.show()

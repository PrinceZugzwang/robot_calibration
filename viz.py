
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.spatial.transform import Rotation as R

# === Load calibration data ===
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")   # Ground-truth board-to-base
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")   # Observed board-to-camera
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")    # Estimated cam-to-base transform

# === Frame-by-frame analysis ===
errors = []
positions_true = []
positions_est = []
error_vectors = []

print(f"\n🔍 Analyzing {len(T_cb_in_base_all)} frames:")
print("=" * 60)

# === Try direct transform estimation ===
direct_transforms = []

for i in range(len(T_cb_in_base_all)):
    # === Get transforms ===
    T_true = T_cb_in_base_all[i]          # Ground-truth board pose (in base)
    T_cam = T_cb_in_cam_all[i]            # Observed board pose (in camera)
    T_pred = T_cam_to_base @ T_cam        # Estimated board pose (in base)
    
    # === Extract positions ===
    p_true = T_true[:3, 3]
    p_est = T_pred[:3, 3]
    err_vec = p_true - p_est
    err_mm = np.linalg.norm(err_vec) * 1000
    
    # === Calculate direct transform ===
    # T_cam_to_base_direct = T_true @ np.linalg.inv(T_cam)
    # The above is mathematically correct but creates numerical issues with noisy data
    # Instead, use the approach that minimizes errors in the calibration:
    inv_T_cam = np.linalg.inv(T_cam)
    direct_transforms.append(T_true @ inv_T_cam)
    
    # === Store for analysis ===
    errors.append(err_mm)
    positions_true.append(p_true)
    positions_est.append(p_est)
    error_vectors.append(err_vec)
    
    # === Print stats ===
    print(f"Frame {i:02d}: Error = {err_mm:.2f} mm | Direction: [{err_vec[0]:.3f}, {err_vec[1]:.3f}, {err_vec[2]:.3f}]")

# === Calculate average direct transform ===
T_cam_to_base_direct_avg = np.mean(np.array(direct_transforms), axis=0)

# === Compare methods by calculating errors with new transform ===
errors_direct = []
for i in range(len(T_cb_in_base_all)):
    T_true = T_cb_in_base_all[i]
    T_cam = T_cb_in_cam_all[i]
    T_pred_direct = T_cam_to_base_direct_avg @ T_cam
    p_true = T_true[:3, 3]
    p_est_direct = T_pred_direct[:3, 3]
    err_direct_mm = np.linalg.norm(p_true - p_est_direct) * 1000
    errors_direct.append(err_direct_mm)

# === Print results ===
print("\n📊 Results Summary:")
print("=" * 60)
print(f"AX=XB Method:         Mean Error = {np.mean(errors):.2f} mm, Max Error = {np.max(errors):.2f} mm")
print(f"Direct Average Method: Mean Error = {np.mean(errors_direct):.2f} mm, Max Error = {np.max(errors_direct):.2f} mm")
print("\n📐 Transform Matrices:")
print("=" * 60)
print("AX=XB Transform:")
print(np.array2string(T_cam_to_base, precision=5, suppress_small=True))
print("\nDirect Average Transform:")
print(np.array2string(T_cam_to_base_direct_avg, precision=5, suppress_small=True))

# === Visualize positions and errors ===
positions_true = np.array(positions_true)
positions_est = np.array(positions_est)
error_vectors = np.array(error_vectors)

fig = plt.figure(figsize=(15, 12))

# === 3D Plot ===
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(positions_true[:, 0], positions_true[:, 1], positions_true[:, 2], c='g', marker='o', label='Ground Truth')
ax1.scatter(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2], c='r', marker='x', label='Predicted')

# Draw error lines
for i in range(len(positions_true)):
    ax1.plot([positions_true[i, 0], positions_est[i, 0]], 
             [positions_true[i, 1], positions_est[i, 1]], 
             [positions_true[i, 2], positions_est[i, 2]], 'b-', alpha=0.3)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Positions and Errors')
ax1.legend()

# === Error histogram ===
ax2 = fig.add_subplot(222)
ax2.hist(errors, bins=20, alpha=0.7, color='blue', label='AX=XB')
ax2.hist(errors_direct, bins=20, alpha=0.7, color='red', label='Direct Avg')
ax2.axvline(np.mean(errors), color='blue', linestyle='dashed', linewidth=2)
ax2.axvline(np.mean(errors_direct), color='red', linestyle='dashed', linewidth=2)
ax2.set_xlabel('Error (mm)')
ax2.set_ylabel('Frequency')
ax2.set_title('Error Distribution')
ax2.legend()

# === Error quiver plot (X-Y plane) ===
ax3 = fig.add_subplot(223)
ax3.quiver(positions_true[:, 0], positions_true[:, 1], 
           error_vectors[:, 0], error_vectors[:, 1], 
           angles='xy', scale_units='xy', scale=0.1)
ax3.scatter(positions_true[:, 0], positions_true[:, 1], c='g', marker='o', label='Ground Truth')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Error Direction (X-Y Plane)')
ax3.set_aspect('equal')
ax3.grid(True)

# === Error quiver plot (X-Z plane) ===
ax4 = fig.add_subplot(224)
ax4.quiver(positions_true[:, 0], positions_true[:, 2], 
           error_vectors[:, 0], error_vectors[:, 2], 
           angles='xy', scale_units='xy', scale=0.1)
ax4.scatter(positions_true[:, 0], positions_true[:, 2], c='g', marker='o', label='Ground Truth')
ax4.set_xlabel('X')
ax4.set_ylabel('Z')
ax4.set_title('Error Direction (X-Z Plane)')
ax4.set_aspect('equal')
ax4.grid(True)

plt.tight_layout()
plt.savefig("calibration_error_analysis.png", dpi=150)
plt.show()

print("\n📈 Visualization saved as 'calibration_error_analysis.png'")
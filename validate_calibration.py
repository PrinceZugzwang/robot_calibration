import numpy as np

# Load poses
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")   # shape (N, 4, 4)
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")   # shape (N, 4, 4)
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")    # shape (4, 4)

# Invert to get T_base_to_cam
T_base_to_cam = np.linalg.inv(T_cam_to_base)

errors = []
for T_cb_base, T_cb_cam in zip(T_cb_in_base_all, T_cb_in_cam_all):
    # Transform the camera-frame checkerboard pose into base frame using T_cam_to_base
    T_cb_est_base = T_cam_to_base @ T_cb_cam

    # Compare it to the ground truth MuJoCo one
    error = np.linalg.norm(T_cb_est_base[:3, 3] - T_cb_base[:3, 3])
    errors.append(error)

print("✅ Checked {} calibration samples.".format(len(errors)))
print("📏 Mean position error (m):", np.mean(errors))
print("📉 Min / Max error:", np.min(errors), "/", np.max(errors))

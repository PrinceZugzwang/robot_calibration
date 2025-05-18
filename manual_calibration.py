import numpy as np

# Load data
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")

# Choose a frame
i = 0  # Use first frame

# Get transforms
T_cb_in_base = T_cb_in_base_all[i]
T_cam_in_cb = T_cb_in_cam_all[i]
T_cb_in_cam = np.linalg.inv(T_cam_in_cb)

# Calculate T_cam_to_base directly for this frame
# T_cb_in_base = T_cam_to_base @ T_cb_in_cam
# Therefore: T_cam_to_base = T_cb_in_base @ inv(T_cb_in_cam)
T_cam_to_base_direct = T_cb_in_base @ np.linalg.inv(T_cb_in_cam)

print("=== MANUAL SINGLE-FRAME CALIBRATION ===")
print("Direct T_cam_to_base from frame", i)
print(T_cam_to_base_direct)

# Test this on a different frame
test_frame = min(i+1, len(T_cb_in_base_all)-1)
T_true = T_cb_in_base_all[test_frame]
T_cb_in_cam_test = np.linalg.inv(T_cb_in_cam_all[test_frame])

# Apply our manually derived transformation
p_origin = np.array([0, 0, 0, 1])
p_base_gt = T_true @ p_origin
p_base_pred = T_cam_to_base_direct @ (T_cb_in_cam_test @ p_origin)
error = np.linalg.norm(p_base_gt[:3] - p_base_pred[:3]) * 1000

print("\nTesting on frame", test_frame)
print(f"Error: {error:.1f} mm")

# Try various coordinate transforms
print("\nTrying coordinate transforms:")
# 90° rotation around Z (common in OpenCV vs other systems)
Rz_90 = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Apply to checkerboard-to-camera transform
T_cb_in_cam_rot = T_cb_in_cam_test @ Rz_90
p_base_pred_rot = T_cam_to_base_direct @ (T_cb_in_cam_rot @ p_origin)
error_rot = np.linalg.norm(p_base_gt[:3] - p_base_pred_rot[:3]) * 1000
print(f"With 90° Z rotation: {error_rot:.1f} mm")
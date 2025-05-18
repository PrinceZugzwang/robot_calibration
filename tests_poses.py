import numpy as np

# Load data
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")

# Pick a frame
i = 0  # Test first frame

# Get ground truth and transforms
T_true = T_cb_in_base_all[i]
T_cam_in_cb = T_cb_in_cam_all[i]
T_cb_in_cam = np.linalg.inv(T_cam_in_cb)

print("=== RAW TRANSFORM COMPARISON ===")
print("T_true (board in base):")
print(T_true)
print("\nT_cb_in_cam (board in camera):")
print(T_cb_in_cam)
print("\nT_cam_to_base (calibration matrix):")
print(T_cam_to_base)

# Direct comparison of values
print("\n=== DIRECT POSE PREDICTION ===")

# Origin in board frame
origin = np.array([0, 0, 0, 1])

# Transform via ground truth
p_base_gt = T_true @ origin
print("Board origin in base via GT:", p_base_gt[:3])

# Transform via camera chain
p_base_pred = T_cam_to_base @ (T_cb_in_cam @ origin)
print("Board origin in base via Camera:", p_base_pred[:3])
print(f"Error: {np.linalg.norm(p_base_gt[:3] - p_base_pred[:3])*1000:.1f} mm")

# Check board position in camera frame
print("\nBoard position in camera frame:", T_cb_in_cam[:3, 3])
print("Board distance from camera:", np.linalg.norm(T_cb_in_cam[:3, 3]), "meters")

# Print camera position in base frame (from calibration)
cam_pos_in_base = T_cam_to_base[:3, 3]
print("\nCamera position in base frame:", cam_pos_in_base)
print("Camera height above base:", cam_pos_in_base[2], "meters")

# Check if Z directions align
print("\n=== AXIS ALIGNMENT CHECK ===")
# Z-axis of board in base frame (from ground truth)
z_board_in_base = T_true[:3, 2]
print("Board Z-axis in base frame (GT):", z_board_in_base)

# Z-axis of board in base frame (via camera)
z_board_in_cam = T_cb_in_cam[:3, 2]
z_board_in_base_pred = T_cam_to_base[:3, :3] @ z_board_in_cam
print("Board Z-axis in base frame (Pred):", z_board_in_base_pred)
print(f"Z-axis alignment error: {np.arccos(np.dot(z_board_in_base, z_board_in_base_pred))*180/np.pi:.1f} degrees")

# Test camera Z-axis
cam_z_in_base = T_cam_to_base[:3, 2]
print("\nCamera Z-axis in base frame:", cam_z_in_base)
print("Is camera pointing down?", cam_z_in_base[2] < 0)
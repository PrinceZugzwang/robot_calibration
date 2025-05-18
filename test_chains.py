import numpy as np

# Load data
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")
T_cb_in_cam_all = np.load("debug_view_fixed/camera_poses.npy")
T_cam_to_base = np.load("debug_view_fixed/T_cam_to_base.npy")

# Test frame
i = 6

# Ground truth board pose
T_true = T_cb_in_base_all[i]
T_cb_in_cam = np.linalg.inv(T_cb_in_cam_all[i])

print("=== TESTING DIFFERENT TRANSFORMATION CHAINS ===")
print(f"Frame: {i}")

# Original transformation
p_origin = np.array([0, 0, 0, 1])
p_base_gt = T_true @ p_origin
p_base_pred = T_cam_to_base @ (T_cb_in_cam @ p_origin)
error = np.linalg.norm(p_base_gt[:3] - p_base_pred[:3]) * 1000
print(f"Original Error: {error:.1f} mm")

# Test axes permutations
permutations = [
    # Rotation matrices for different permutations
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Identity
    np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Flip X
    np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Flip Y
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),  # Flip Z
    np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Swap X/Y
    np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),  # Swap X/Z
    np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),  # Swap Y/Z
    np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Rotate 90° around Z
    np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Rotate -90° around Z
    np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),  # Rotate 90° around X
    np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])   # Rotate 90° around Y
]

perm_names = [
    "Identity", "Flip X", "Flip Y", "Flip Z", "Swap X/Y", 
    "Swap X/Z", "Swap Y/Z", "Rot 90° Z", "Rot -90° Z", "Rot 90° X", "Rot 90° Y"
]

# Test all permutations
best_error = error
best_perm = "Original"
best_idx = -1

print("\n=== PERMUTATION TEST RESULTS ===")
for idx, (perm, name) in enumerate(zip(permutations, perm_names)):
    # Apply permutation to checkerboard-to-camera transform
    T_cb_in_cam_perm = T_cb_in_cam @ perm
    
    # Transform to base frame
    p_base_perm = T_cam_to_base @ (T_cb_in_cam_perm @ p_origin)
    
    # Calculate error
    error_perm = np.linalg.norm(p_base_gt[:3] - p_base_perm[:3]) * 1000
    print(f"{name}: {error_perm:.1f} mm")
    
    if error_perm < best_error:
        best_error = error_perm
        best_perm = name
        best_idx = idx

print(f"\nBest permutation: {best_perm} with error {best_error:.1f} mm")

if best_idx >= 0:
    print("\n=== BEST PERMUTATION MATRIX ===")
    print(permutations[best_idx])
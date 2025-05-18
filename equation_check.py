import numpy as np
from scipy.spatial.transform import Rotation as R

# Import the solve_AX_XB function from your calibration_capture.py
from test2 import solve_AX_XB

print("=== TESTING AX=XB SOLUTION ALGORITHM ===")

# Create a known X transform (e.g., translation + rotation)
T_X_known = np.eye(4)
T_X_known[:3,3] = [0.1, 0.2, 0.3]  # Known translation
T_X_known[:3,:3] = R.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix() # Known rotation

print("Known X transform:")
print(T_X_known)

# Create matching A and B pairs that satisfy AX = XB
A_test = []
B_test = []

# Generate several test cases
for i in range(5):
    # Create random A transformation
    T_A = np.eye(4)
    T_A[:3,3] = np.random.rand(3) * 0.5  # Random translation
    T_A[:3,:3] = R.random().as_matrix()  # Random rotation
    
    # Calculate the corresponding B that satisfies AX = XB
    # If AX = XB, then B = X^(-1) A X
    T_B = np.linalg.inv(T_X_known) @ T_A @ T_X_known
    
    A_test.append(T_A)
    B_test.append(T_B)
    
    print(f"\nTest Pair {i+1}:")
    print(f"A{i+1}:\n{T_A}")
    print(f"B{i+1}:\n{T_B}")
    
    # Verify the equation: A_i * X = X * B_i
    left_side = T_A @ T_X_known
    right_side = T_X_known @ T_B
    error = np.linalg.norm(left_side - right_side)
    print(f"Verification Error: {error:.10f} (should be near zero)")

# Solve using your algorithm
T_X_solved = solve_AX_XB(A_test, B_test)

print("\nTest Results:")
print("Known X transform:")
print(T_X_known)
print("\nSolved X transform:")
print(T_X_solved)

# Calculate error metrics
trans_error = np.linalg.norm(T_X_known[:3,3] - T_X_solved[:3,3])
rot_error = np.linalg.norm(T_X_known[:3,:3] - T_X_solved[:3,:3], ord='fro')
total_error = np.linalg.norm(T_X_known - T_X_solved, ord='fro')

print(f"\nError Metrics:")
print(f"Translation Error: {trans_error*1000:.3f} mm")
print(f"Rotation Error: {rot_error:.6f}")
print(f"Total Frobenius Norm Error: {total_error:.6f}")

# Test with noisy data (more realistic)
print("\n=== TESTING WITH NOISE ===")
A_noisy = []
B_noisy = []

for i in range(len(A_test)):
    # Add small noise to translation (few mm)
    T_A_noisy = A_test[i].copy()
    T_A_noisy[:3,3] += np.random.normal(0, 0.003, 3)  # ~3mm noise
    
    # Add small noise to rotation (few degrees)
    noise_rot = R.from_euler('xyz', np.random.normal(0, 1, 3), degrees=True).as_matrix()
    T_A_noisy[:3,:3] = noise_rot @ T_A_noisy[:3,:3]
    
    # Calculate the corresponding B with the same noise level
    T_B_noisy = B_test[i].copy()
    T_B_noisy[:3,3] += np.random.normal(0, 0.003, 3)
    noise_rot = R.from_euler('xyz', np.random.normal(0, 1, 3), degrees=True).as_matrix()
    T_B_noisy[:3,:3] = noise_rot @ T_B_noisy[:3,:3]
    
    A_noisy.append(T_A_noisy)
    B_noisy.append(T_B_noisy)

# Solve using your algorithm with noisy data
T_X_noisy = solve_AX_XB(A_noisy, B_noisy)

print("Solved X transform with noisy data:")
print(T_X_noisy)

# Calculate error metrics
trans_error_noisy = np.linalg.norm(T_X_known[:3,3] - T_X_noisy[:3,3])
rot_error_noisy = np.linalg.norm(T_X_known[:3,:3] - T_X_noisy[:3,:3], ord='fro')
total_error_noisy = np.linalg.norm(T_X_known - T_X_noisy, ord='fro')

print(f"\nError Metrics with Noise:")
print(f"Translation Error: {trans_error_noisy*1000:.3f} mm")
print(f"Rotation Error: {rot_error_noisy:.6f}")
print(f"Total Frobenius Norm Error: {total_error_noisy:.6f}")
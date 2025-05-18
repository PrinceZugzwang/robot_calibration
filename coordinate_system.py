import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load model and data ===
model = mujoco.MjModel.from_xml_path("combined.xml")
data = mujoco.MjData(model)

# === Function to visualize coordinate frames ===
def visualize_coordinate_frames():
    # Initialize forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get camera position and orientation
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    
    # Get checkerboard position and orientation
    board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "checkerboard_mount")
    board_pos = data.xpos[board_id]
    board_mat = data.xmat[board_id].reshape(3, 3)
    
    # Extract axes
    cam_x = cam_mat[:, 0] * 0.1
    cam_y = cam_mat[:, 1] * 0.1
    cam_z = cam_mat[:, 2] * 0.1
    
    board_x = board_mat[:, 0] * 0.1
    board_y = board_mat[:, 1] * 0.1
    board_z = board_mat[:, 2] * 0.1
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera frame
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              cam_x[0], cam_x[1], cam_x[2], color='r', label='Camera X')
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              cam_y[0], cam_y[1], cam_y[2], color='g', label='Camera Y')
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
              cam_z[0], cam_z[1], cam_z[2], color='b', label='Camera Z')
    
    # Plot checkerboard frame
    ax.quiver(board_pos[0], board_pos[1], board_pos[2], 
              board_x[0], board_x[1], board_x[2], color='darkred', label='Board X')
    ax.quiver(board_pos[0], board_pos[1], board_pos[2], 
              board_y[0], board_y[1], board_y[2], color='darkgreen', label='Board Y')
    ax.quiver(board_pos[0], board_pos[1], board_pos[2], 
              board_z[0], board_z[1], board_z[2], color='darkblue', label='Board Z')
    
    # Plot origin
    ax.scatter(0, 0, 0, color='black', s=100, label='Origin')
    
    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('MuJoCo Coordinate Frames')
    ax.legend()
    
    # Make equal aspect ratio
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    plt.savefig("coordinate_frames.png", dpi=150)
    plt.show()
    
    print("✅ Saved coordinate frame visualization to 'coordinate_frames.png'")
    
    # Print numerical values
    print("\n📊 Coordinate Frame Data:")
    print("=" * 60)
    print(f"Camera Position: {cam_pos}")
    print("Camera Orientation:")
    print(f"  X-axis: {cam_x / np.linalg.norm(cam_x)}")
    print(f"  Y-axis: {cam_y / np.linalg.norm(cam_y)}")
    print(f"  Z-axis: {cam_z / np.linalg.norm(cam_z)}")
    print("\nCheckerboard Position:", board_pos)
    print("Checkerboard Orientation:")
    print(f"  X-axis: {board_x / np.linalg.norm(board_x)}")
    print(f"  Y-axis: {board_y / np.linalg.norm(board_y)}")
    print(f"  Z-axis: {board_z / np.linalg.norm(board_z)}")

# === Check camera intrinsics ===
def verify_camera_intrinsics():
    # MuJoCo camera data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
    fovy = model.cam_fovy[cam_id]
    resolution = (640, 480)  # From your code
    
    # Calculate intrinsics using your method
    def get_intrinsics(fov_deg, w, h):
        f = (w / 2) / np.tan(np.deg2rad(fov_deg / 2))
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    
    K_yours = get_intrinsics(80, *resolution)  # From your code
    
    # Calculate intrinsics directly from MuJoCo's fovy
    # Note: MuJoCo uses vertical FOV, not horizontal
    f_mujoco = (resolution[1] / 2) / np.tan(np.deg2rad(fovy / 2))
    K_mujoco = np.array([
        [f_mujoco * resolution[0]/resolution[1], 0, resolution[0] / 2],
        [0, f_mujoco, resolution[1] / 2],
        [0, 0, 1]
    ])
    
    print("\n📷 Camera Intrinsics Comparison:")
    print("=" * 60)
    print(f"MuJoCo fovy: {fovy} degrees")
    print(f"Your FOV: 80 degrees")
    print("\nYour intrinsics matrix:")
    print(np.array2string(K_yours, precision=2))
    print("\nMuJoCo-derived intrinsics matrix:")
    print(np.array2string(K_mujoco, precision=2))
    print("\nDifference (yours - MuJoCo):")
    diff = K_yours - K_mujoco
    print(np.array2string(diff, precision=2))
    print(f"Max absolute difference: {np.max(np.abs(diff)):.2f}")

# === Run the checks ===
mujoco.mj_forward(model, data)
visualize_coordinate_frames()
verify_camera_intrinsics()

# === Suggest a fix for AX=XB formulation ===
print("\n🛠️ Suggested AX=XB Fix:")
print("=" * 60)
print("Based on your error patterns, try this AX=XB formulation:")
print("""
# Instead of:
poses.append((T_cam_in_cb, T_base_in_cb))

# Try this (original transforms - no inverses):
poses.append((T_cb, T_base_board))

# Or even try:
A_transforms = []
B_transforms = []
for i in range(len(T_cb_list) - 1):
    A = np.linalg.inv(T_cb_list[i]) @ T_cb_list[i+1]  # Motion in camera frame
    B = np.linalg.inv(T_base_board_list[i]) @ T_base_board_list[i+1]  # Motion in base frame
    A_transforms.append(A)
    B_transforms.append(B)
""")
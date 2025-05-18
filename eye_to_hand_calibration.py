import mujoco
import numpy as np
import cv2
import time
import os
import json
from scipy.spatial.transform import Rotation as R

# --- Config ---
model_path = "combined.xml"
camera_name = "fixed_cam"               # ✅ Fixed overhead camera
board_body_name = "checkerboard_mount"  # ✅ Checkerboard now moves
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
resolution = (640, 480)
fov = 80
pattern_size = (4, 6)       # inner corners
square_size = 0.025
brightness_thresh = 20
save_dir = "debug_view_fixed"
os.makedirs(save_dir, exist_ok=True)

# --- Camera intrinsics ---
def get_intrinsics(fov_deg, w, h):
    f = (w / 2) / np.tan(np.deg2rad(fov_deg / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

K = get_intrinsics(fov, *resolution)

# --- Pose Helpers ---
def get_pose(model, data, body):
    bid = model.body(body).id
    return data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy()

def pose_to_SE3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# --- MuJoCo ---
def render_image(model, data, cam_name):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer = mujoco.Renderer(model, *resolution)
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()

def randomize_joints(model, data):
    for name in joint_names:
        jid = model.joint(name).id
        low, high = model.jnt_range[jid]
        data.qpos[jid] = np.random.uniform(low, high)
    mujoco.mj_forward(model, data)

# --- Checkerboard Detection ---
def estimate_checkerboard_pose(img, pattern_size, square_size, K):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not ret:
        return None

    # Refine corners for better accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Create object points with the correct pattern orientation
    # OpenCV expects checkerboard with z=0 plane
    # Switch from (row, col) to (x, y) coordinates
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0]):
            objp[i*pattern_size[0] + j, 0] = j * square_size
            objp[i*pattern_size[0] + j, 1] = i * square_size
    
    # Use SOLVEPNP_ITERATIVE for better accuracy
    flags = cv2.SOLVEPNP_ITERATIVE
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, None, flags=flags)
    
    # Calculate reprojection error to check quality
    proj_points, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
    error = np.sqrt(np.sum((corners - proj_points)**2, axis=2)).mean()
    print(f"Reprojection error: {error:.4f} pixels")
    
    # Only accept poses with reasonable reprojection error
    if error > 2.0:  # More than 2 pixels error
        print("Warning: High reprojection error, pose may be unreliable")
    
    R_mat, _ = cv2.Rodrigues(rvec)
    return pose_to_SE3(R_mat, tvec.flatten())

# --- Calibration AX=XB ---
def compute_relative_transforms(pairs):
    A, B = [], []
    for i in range(len(pairs) - 1):
        A.append(np.linalg.inv(pairs[i][0]) @ pairs[i + 1][0])
        B.append(np.linalg.inv(pairs[i][1]) @ pairs[i + 1][1])
    return A, B

def solve_AX_XB(A_list, B_list):
    n = len(A_list)
    M = np.zeros((3 * n, 3))
    b = np.zeros((3 * n, 1))
    for i in range(n):
        Ra, Rb = A_list[i][:3, :3], B_list[i][:3, :3]
        alpha = R.from_matrix(Ra).as_rotvec()
        beta = R.from_matrix(Rb).as_rotvec()
        M[3*i:3*i+3, :] = np.eye(3) - Rb
        b[3*i:3*i+3] = (beta - alpha).reshape(3,1)
    omega = np.linalg.lstsq(M, b, rcond=None)[0]
    R_x = R.from_rotvec(omega.flatten()).as_matrix()
    C, d = [], []
    for i in range(n):
        ta = A_list[i][:3,3]
        tb = B_list[i][:3,3]
        Rb = B_list[i][:3,:3]
        C.append(Rb - np.eye(3))
        d.append((ta - R_x @ tb).reshape(3,1))
    t_x = np.linalg.lstsq(np.vstack(C), np.vstack(d), rcond=None)[0]
    T = np.eye(4)
    T[:3,:3] = R_x
    T[:3,3] = t_x.flatten()
    return T

# --- MAIN LOOP ---
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

poses = []
board_poses = []
images_saved = 0
frame_idx = 0

print("📸 Eye-to-Hand: Capturing frames. Ctrl+C to stop.")

try:
    while True:
        randomize_joints(model, data)
        img = render_image(model, data, camera_name)
        brightness = np.mean(img)

        if brightness > brightness_thresh:
            cv2.imwrite(os.path.join(save_dir, f"bright_frame_{frame_idx:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if brightness < brightness_thresh:
            frame_idx += 1
            continue

        T_cb = estimate_checkerboard_pose(img, pattern_size, square_size, K)
        if T_cb is None:
            frame_idx += 1
            continue

        R_board, t_board = get_pose(model, data, board_body_name)
        T_base_board = pose_to_SE3(R_board, t_board)

        # ✅ Invert both to get camera and base poses in checkerboard frame
        T_cam_in_cb = np.linalg.inv(T_cb)
        T_base_in_cb = np.linalg.inv(T_base_board)

        # ✅ Append the corrected pair
        poses.append((T_cam_in_cb, T_base_in_cb))
        board_poses.append(T_base_board)

        # ✅ Save the raw data (no need to save inverted for now)
        np.save(os.path.join(save_dir, "board_poses.npy"), np.array(board_poses))
        np.save(os.path.join(save_dir, "camera_poses.npy"), np.array([p[0] for p in poses]))

        cv2.imwrite(os.path.join(save_dir, f"good_frame_{images_saved:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved {images_saved} | Brightness {brightness:.1f}")
        images_saved += 1
        frame_idx += 1

except KeyboardInterrupt:
    print("\n🛑 Sampling interrupted.")

finally:
    print(f"\n📦 Collected {len(poses)} usable frames.")
    if len(poses) >= 2:
        A, B = compute_relative_transforms(poses)
        T = solve_AX_XB(A, B)
        np.save(os.path.join(save_dir, "T_cam_to_base.npy"), T)
        print("📏 Calibration result (Eye-to-Hand: T_cam_to_base):\n", T)
    else:
        print("❌ Not enough data for calibration.")
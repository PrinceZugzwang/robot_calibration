import mujoco
import numpy as np
import cv2
import os
import time
from scipy.spatial.transform import Rotation as R
from glob import glob

# ------------ Config ------------
model_path = "combined.xml"
camera_name = "ee_cam"
ee_body_name = "link6"
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
pattern_size = (6, 9)
square_size = 0.025  # in meters
resolution = (640, 480)
fov = 45
save_dir = "debug_view"
num_required_poses = 15
brightness_thresh = 20
# --------------------------------

os.makedirs(save_dir, exist_ok=True)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, *resolution)
cam_id = model.camera(camera_name).id
cv2.namedWindow("EE Camera View", cv2.WINDOW_NORMAL)

ee_poses = []
pose_pairs = []

def randomize_joints():
    for name in joint_names:
        jid = model.joint(name).id
        low, high = model.jnt_range[jid]
        mid = (low + high) / 2
        span = (high - low) * 0.3
        data.qpos[jid] = np.random.uniform(mid - span/2, mid + span/2)
    mujoco.mj_forward(model, data)

def get_pose(body_name):
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()
    return mat, pos

def pose_to_SE3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_camera_intrinsics(fov_deg, width, height):
    fov_rad = np.deg2rad(fov_deg)
    f = (width / 2) / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K

def estimate_checkerboard_pose(img, pattern_size, square_size, K):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not ret:
        print("❌ Checkerboard NOT detected")
        return None

    # Optional subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
    R_mat, _ = cv2.Rodrigues(rvec)
    return pose_to_SE3(R_mat, tvec.flatten())


def compute_relative_transforms(poses):
    A_list, B_list = [], []
    for i in range(len(poses) - 1):
        A = np.linalg.inv(poses[i][0]) @ poses[i + 1][0]
        B = np.linalg.inv(poses[i][1]) @ poses[i + 1][1]
        A_list.append(A)
        B_list.append(B)
    return A_list, B_list

def solve_AX_XB(A_list, B_list):
    n = len(A_list)
    M = np.zeros((3 * n, 3))
    b = np.zeros((3 * n, 1))
    for i in range(n):
        Ra = A_list[i][:3, :3]
        Rb = B_list[i][:3, :3]
        alpha = R.from_matrix(Ra).as_rotvec()
        beta = R.from_matrix(Rb).as_rotvec()
        M[3*i:3*i+3, :] = np.eye(3) - Rb
        b[3*i:3*i+3, 0] = beta - alpha

    omega = np.linalg.lstsq(M, b, rcond=None)[0]
    theta = np.linalg.norm(omega)
    r = omega.flatten() / theta if theta > 1e-5 else omega.flatten()
    R_x = R.from_rotvec(r * theta).as_matrix()

    C, d = [], []
    for i in range(n):
        ta = A_list[i][:3, 3]
        tb = B_list[i][:3, 3]
        Rb = B_list[i][:3, :3]
        C_i = Rb - np.eye(3)
        d_i = (ta - R_x @ tb).reshape(3, 1)
        for row in range(3):
            C.append(C_i[row:row + 1, :])
            d.append(d_i[row:row + 1])
    C = np.vstack(C)
    d = np.vstack(d)
    t_x = np.linalg.lstsq(C, d, rcond=None)[0]

    T_x = np.eye(4)
    T_x[:3, :3] = R_x
    T_x[:3, 3] = t_x.flatten()
    return T_x

# ---------------- MAIN LOOP ------------------
K = get_camera_intrinsics(fov, *resolution)
good_count = 0
frame_idx = 0

print("🎥 Capturing until 15 good frames (brightness only)...")

while good_count < num_required_poses:
    randomize_joints()
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mean_brightness = np.mean(img)

    R_ee, t_ee = get_pose(ee_body_name)
    T_ee = pose_to_SE3(R_ee, t_ee)

    if mean_brightness > brightness_thresh:
        save_path = os.path.join(save_dir, f"good_frame_{frame_idx:03d}.png")
        cv2.imwrite(save_path, img)
        ee_poses.append(T_ee)

        T_cb = estimate_checkerboard_pose(img, pattern_size, square_size, K)
        if T_cb is not None:
            pose_pairs.append((T_cb, T_ee))
            good_count += 1
            print(f"[{frame_idx}] ✅ Saved (brightness {mean_brightness:.1f})")
        else:
            print(f"[{frame_idx}] ⚠️ Bright but no checkerboard found")
    else:
        print(f"[{frame_idx}] ❌ Too dark (brightness {mean_brightness:.1f})")

    cv2.imshow("EE Camera View", img)
    key = cv2.waitKey(100)
    if key == 27:
        break

    frame_idx += 1

cv2.destroyAllWindows()

# ---------------- Calibration ------------------
print(f"\n🧠 Running calibration on {len(pose_pairs)} poses...")

A_list, B_list = compute_relative_transforms(pose_pairs)
T_cam_ee = solve_AX_XB(A_list, B_list)

print("\n✅ Calibration complete!")
np.set_printoptions(precision=4, suppress=True)
print("Estimated T_cam_ee:\n", T_cam_ee)

# Save EE poses
np.save(os.path.join(save_dir, "ee_poses.npy"), np.array(ee_poses))

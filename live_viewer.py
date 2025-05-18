import mujoco
import mujoco.viewer
import numpy as np
import time

# === Load model and data ===
model = mujoco.MjModel.from_xml_path("combined.xml")
data = mujoco.MjData(model)

# === Load calibration data ===
T_cb_in_base_all = np.load("debug_view_fixed/board_poses.npy")   # Ground-truth board-to-base
T_cb_in_cam_all  = np.load("debug_view_fixed/camera_poses.npy")  # Observed board-to-camera
T_cam_to_base    = np.load("debug_view_fixed/T_cam_to_base.npy") # Estimated cam-to-base transform

# === End-effector body name ===
ee_body_name = "link6"

# === Get mocap body IDs for GT and predicted marker ===
mocap_true_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gt_marker_body")
mocap_est_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pred_marker_body")

# === Move robot to a 3D target (slowly) ===
def move_to_target_slow(model, data, targetpos, viewer, steps=100, tol=1e-4):
    for i in range(steps):
        mujoco.mj_forward(model, data)
        ee_bid = model.body(ee_body_name).id
        ee_pos = data.xpos[ee_bid]
        err = targetpos - ee_pos
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, None, ee_bid)
        dq = np.linalg.pinv(jacp[:, :model.nv]) @ (err / 3.0)
        data.qpos[:model.nv] += dq
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.05)

# === MAIN VISUALIZATION LOOP ===
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("🎥 Starting visualization...")

    for i in range(len(T_cb_in_base_all)):
        # === Get transforms
        T_true = T_cb_in_base_all[i]           # Ground-truth board pose (in base)
        T_cb_in_cam = np.linalg.inv(T_cb_in_cam_all[i])  # Convert back from T_cam_in_cb
        T_pred = T_cam_to_base @ T_cb_in_cam   # Now correct! 
        # === Extract positions
        p_true = T_true[:3, 3]
        p_est  = T_pred[:3, 3]
        err_mm = np.linalg.norm(p_true - p_est) * 1000
        print(f"[Frame {i}] 🟢 GT: {p_true}, 🔴 Pred: {p_est}, Error: {err_mm:.1f} mm")

        # === Update mocap body positions (key fix) ===
        data.mocap_pos[mocap_true_id - model.nmocap] = p_true
        data.mocap_pos[mocap_est_id - model.nmocap] = p_est

        # === Also update rotations if needed ===
        # data.mocap_quat[mocap_true_id - model.nmocap] = ... # Convert rotation matrix to quaternion if needed
        # data.mocap_quat[mocap_est_id - model.nmocap] = ... # Convert rotation matrix to quaternion if needed

        # === Recompute positions (so sites are drawn correctly)
        mujoco.mj_forward(model, data)

        # === Move to predicted marker position
        move_to_target_slow(model, data, p_est, viewer)

        # === Pause to observe
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.03)


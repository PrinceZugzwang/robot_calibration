import mujoco
import numpy as np
import cv2
import os
import time

model_path = "combined.xml"
camera_name = "ee_cam"
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
resolution = (640, 480)
save_dir = "debug_view"

os.makedirs(save_dir, exist_ok=True)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, *resolution)
cam_id = model.camera(camera_name).id

cv2.namedWindow("EE Camera View", cv2.WINDOW_NORMAL)

def randomize_joints():
    for name in joint_names:
        jid = model.joint(name).id
        low, high = model.jnt_range[jid]
        mid = (low + high) / 2
        span = (high - low) * 0.3
        data.qpos[jid] = np.random.uniform(mid - span/2, mid + span/2)
    mujoco.mj_forward(model, data)

frame_idx = 0
print("🎥 Starting debug viewer (filter: brightness > 20). Press ESC to quit.\n")

while True:
    randomize_joints()

    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    brightness = np.mean(img)
    is_good = brightness > 20

    tag = "✅" if is_good else "❌"
    print(f"[Frame {frame_idx}] {tag} Brightness: {brightness:.2f}")

    if is_good:
        cv2.imwrite(os.path.join(save_dir, f"good_frame_{frame_idx:03d}.png"), img)

    cv2.imshow("EE Camera View", img)
    if cv2.waitKey(300) == 27:
        break

    frame_idx += 1

cv2.destroyAllWindows()

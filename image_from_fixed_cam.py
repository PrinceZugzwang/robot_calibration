import mujoco
import numpy as np
import cv2
import os

# --- Config ---
model_path = "combined.xml"
camera_name = "fixed_cam"
resolution = (640, 480)
save_path = "test_fixed_cam.png"

# --- Load and Setup ---
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, *resolution)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

# --- Advance a few steps for lighting to take effect ---
for _ in range(10):
    mujoco.mj_step(model, data)

# --- Render ---
renderer.update_scene(data, camera=cam_id)
img = renderer.render()

# --- Brightness Check ---
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
mean_brightness = np.mean(gray)
print(f"🧪 Mean brightness: {mean_brightness:.2f}")
if mean_brightness < 5:
    print("⚠️ Too dark! Possibly occluded or lighting issue.")
else:
    print("✅ Rendering OK!")

# --- Save and View ---
cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"📸 Saved to {save_path}")
cv2.imshow("View", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

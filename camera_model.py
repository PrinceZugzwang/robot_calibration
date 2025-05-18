import numpy as np
import mujoco
import cv2

# Load the model
model = mujoco.MjModel.from_xml_path("combined.xml")
data = mujoco.MjData(model)

# Get the camera ID
cam_name = "fixed_cam"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

print("=== CAMERA MODEL COMPARISON ===")
# MuJoCo camera parameters
print("MuJoCo Camera:")
print(f"Position: {data.cam_xpos[cam_id]}")
print(f"Orientation: {data.cam_xmat[cam_id].reshape(3,3)}")
print(f"FOV (vertical): {model.cam_fovy[cam_id]} degrees")

# Resolution
width, height = 640, 480
aspect = width / height

# Calculate vertical FOV from horizontal and vice versa
fovy = model.cam_fovy[cam_id]
fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy/2)) * aspect) * 180 / np.pi

print(f"Calculated horizontal FOV: {fovx:.2f} degrees")

# OpenCV model parameters using vertical FOV
f_vert = (height/2) / np.tan(np.deg2rad(fovy/2))
f_horz = (width/2) / np.tan(np.deg2rad(fovx/2))

print("\nOpenCV Camera Intrinsics:")
K_vert = np.array([[f_vert, 0, width/2], [0, f_vert, height/2], [0, 0, 1]])
K_horz = np.array([[f_horz, 0, width/2], [0, f_horz, height/2], [0, 0, 1]])

print("Using vertical FOV:")
print(K_vert)
print("\nUsing horizontal FOV:")
print(K_horz)

# Test with a known 3D point projection
test_point = np.array([0.1, 0.1, 0.5])  # 10cm to the right and up, 50cm in front

# MuJoCo way (pseudo-code, would need actual rendering to be precise)
print("\nTest point projection:")
print(f"3D point: {test_point}")

# Calculate expected projection using OpenCV model
test_point_cv = np.array([test_point[0], test_point[1], test_point[2]])
pixel_vert = K_vert @ test_point_cv / test_point_cv[2]
pixel_horz = K_horz @ test_point_cv / test_point_cv[2]

print(f"Projected pixel (vertical FOV): {pixel_vert[:2]}")
print(f"Projected pixel (horizontal FOV): {pixel_horz[:2]}")
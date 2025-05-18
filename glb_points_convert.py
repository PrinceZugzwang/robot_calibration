import trimesh
import numpy as np

# Load the scene
scene = trimesh.load("/home/ubuntu/3d_proj/glbscene (1).glb", force='scene')

# Extract the geometry (it's an OrderedDict)
pc = list(scene.geometry.values())[0]   # first (and only) geometry

print(pc)  # should say PointCloud(vertices.shape=(307200, 3))

# Now get the vertices
points_camera = np.asarray(pc.vertices)  # (N, 3)

# ✅ Now apply your transform
T_cam_to_base = np.load("/home/ubuntu/3d_proj/debug_view_fixed/T_cam_to_base.npy")  # your calibration matrix

# Convert to homogeneous points (N, 4)
points_h = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])

# Apply the transformation
points_base_h = (T_cam_to_base @ points_h.T).T

# Extract (x, y, z)
points_base = points_base_h[:, :3]

# # Save the points in robot base frame
# np.save("points_in_base.npy", points_base)

# print(f"✅ Saved {points_base.shape[0]} points to robot base frame!")

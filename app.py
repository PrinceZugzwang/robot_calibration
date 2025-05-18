import open3d as o3d
import numpy as np
import os
import json

# -------- CONFIG --------
glb_dir = "glb_files/"  # Put your .glb files here
ply_dir = "mast3r_output/"  # Output directory for .ply files
os.makedirs(ply_dir, exist_ok=True)

# Convert .glb to .ply
def convert_glb_to_ply(glb_path, ply_path, sample_points=100000):
    mesh = o3d.io.read_triangle_mesh(glb_path)
    if mesh.is_empty():
        print(f"❌ Could not read {glb_path}")
        return None
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ Converted {glb_path} → {ply_path}")
    return ply_path

# -------- CONVERSION STAGE --------
glb_to_ply_map = {
    "cam0": ("cam0_scene.glb", "cam0.ply"),
    "cam1": ("cam1_scene.glb", "cam1.ply"),
    "cam2": ("cam2_scene.glb", "cam2.ply")
}

ply_paths = {}
for cam, (glb_name, ply_name) in glb_to_ply_map.items():
    glb_path = os.path.join(glb_dir, glb_name)
    ply_path = os.path.join(ply_dir, ply_name)
    converted = convert_glb_to_ply(glb_path, ply_path)
    if converted:
        ply_paths[cam] = ply_path

# -------- LOAD POINT CLOUDS --------
pcd_cam0 = o3d.io.read_point_cloud(ply_paths["cam0"])
pcd_cam1 = o3d.io.read_point_cloud(ply_paths["cam1"])
pcd_cam2 = o3d.io.read_point_cloud(ply_paths["cam2"])

# -------- ALIGN cam1 → cam0 --------
reg_icp_1 = o3d.pipelines.registration.registration_icp(
    source=pcd_cam1,
    target=pcd_cam0,
    max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
T_cam1_to_cam0 = reg_icp_1.transformation

# -------- ALIGN cam2 → cam0 --------
reg_icp_2 = o3d.pipelines.registration.registration_icp(
    source=pcd_cam2,
    target=pcd_cam0,
    max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
T_cam2_to_cam0 = reg_icp_2.transformation

# -------- SAVE TRANSFORM MATRICES --------
extrinsics = {
    "cam0": np.eye(4).tolist(),
    "cam1": T_cam1_to_cam0.tolist(),
    "cam2": T_cam2_to_cam0.tolist()
}
with open("extrinsics_config.json", "w") as f:
    json.dump(extrinsics, f, indent=2)
print("📦 Saved extrinsics_config.json")

# -------- VISUALIZE RESULT --------
pcd_cam1.transform(T_cam1_to_cam0)
pcd_cam2.transform(T_cam2_to_cam0)

o3d.visualization.draw_geometries([pcd_cam0, pcd_cam1, pcd_cam2])

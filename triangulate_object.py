import numpy as np
import json
import cv2

# ---------- CONFIG ----------
EXTRINSICS_PATH = "extrinsics_config.json"
PIXEL_COORDS_PATH = "object_pixel_coords.json"

# Approximate intrinsics (since we skipped true calibration)
# [fx, 0, cx], [0, fy, cy], [0, 0, 1]
K = np.array([
    [1500, 0, 1280],  # fx, 0, cx
    [0, 1500, 720],   # 0, fy, cy
    [0, 0, 1]
])

def pixel_to_ray(K, pixel):
    """Convert 2D pixel to 3D direction vector in camera coordinates."""
    pixel_homog = np.array([pixel[0], pixel[1], 1.0])  # make homogeneous
    ray_cam = np.linalg.inv(K).dot(pixel_homog)
    ray_cam /= np.linalg.norm(ray_cam)
    return ray_cam

def transform_ray(ray, transform):
    """Transform ray from local camera coords to world coords using extrinsics."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    ray_world = R @ ray  # rotate direction
    origin_world = t     # camera center in world
    return origin_world, ray_world

def closest_point_among_rays(origins, directions):
    """Estimate 3D point that minimizes distance to all rays (least squares)."""
    A = []
    b = []
    for o, d in zip(origins, directions):
        d = d / np.linalg.norm(d)
        I = np.eye(3)
        A.append(I - np.outer(d, d))
        b.append((I - np.outer(d, d)) @ o)

    A = np.concatenate(A)
    b = np.concatenate(b)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x

def main():
    # Load extrinsics
    with open(EXTRINSICS_PATH, "r") as f:
        extrinsics = json.load(f)

    # Load clicked 2D points
    with open(PIXEL_COORDS_PATH, "r") as f:
        pixel_coords = json.load(f)

    origins = []
    directions = []

    for cam_name, pixel in pixel_coords.items():
        transform = np.array(extrinsics[cam_name])
        ray_cam = pixel_to_ray(K, pixel)
        origin_world, ray_world = transform_ray(ray_cam, transform)
        origins.append(origin_world)
        directions.append(ray_world)

    object_pos = closest_point_among_rays(origins, directions)
    print(f"\n🎯 Triangulated object position (world frame):\n{object_pos}")

if __name__ == "__main__":
    main()

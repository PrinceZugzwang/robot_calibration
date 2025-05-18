import numpy as np

def get_camera_intrinsics(fov_deg, width, height):
    fov_rad = np.deg2rad(fov_deg)
    f = (width / 2) / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    return K

K = get_camera_intrinsics(fov_deg=80, width=640, height=480)
print(K)

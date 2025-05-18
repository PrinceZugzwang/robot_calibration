import cv2
import os
import numpy as np

# CONFIG
folder = "debug_view"
pattern_size = (4, 6)  # inner corners
square_size = 0.025  # optional
visualize = False

# List all bright frame images
images = sorted([f for f in os.listdir(folder) if f.startswith("good_frame") and f.endswith(".png")])

if not images:
    print("No bright_frame images found.")
    exit()

print(f"🔍 Found {len(images)} bright frames to check.\n")

for fname in images:
    path = os.path.join(folder, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"❌ Failed to read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    


    # Use robust flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    status = "✅ FOUND" if found else "❌ NOT FOUND"
    print(f"[{fname}] {status}")

    if visualize:
        vis_img = img.copy()
        if found:
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, found)
        cv2.imshow("Checkerboard Debug", vis_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit early
            break

cv2.destroyAllWindows()

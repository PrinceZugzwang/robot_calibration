import cv2
import numpy as np
import os

# ---------- CONFIG ----------
image_path = "/home/ubuntu/3d_proj/debug_view/bright_frame_380.png"
pattern_size = (6, 5)  # inner corners (NOT squares!)
square_size_mm = 25    # optional - not needed here

visualize = True
save_debug = True
out_dir = "debug_output"
os.makedirs(out_dir, exist_ok=True)
# ----------------------------

def try_detection(name, img, use_sb=False):
    if use_sb:
        ret, corners = cv2.findChessboardCornersSB(img, pattern_size)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(img, pattern_size, flags)

    print(f"[{name}] {'✅ FOUND' if ret else '❌ NOT FOUND'}")

    if visualize or save_debug:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if ret:
            if use_sb:
                cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
            else:
                # Refine for classic
                corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vis, pattern_size, corners, ret)

        if visualize:
            cv2.imshow(name, vis)
            cv2.waitKey(0)

        if save_debug:
            out_file = os.path.join(out_dir, f"{name.replace(' ', '_')}.png")
            cv2.imwrite(out_file, vis)

# ----- Load image -----
bgr = cv2.imread(image_path)
if bgr is None:
    print("❌ Could not load image:", image_path)
    exit()

gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print("📷 Image loaded:", image_path)
print("📏 Resolution:", gray.shape)

# Brightness debug
print("💡 Brightness → min:", np.min(gray), "max:", np.max(gray), "mean:", np.mean(gray))

# ---- Tests ----
try_detection("Original (classic)", gray, use_sb=False)
try_detection("Original (SB)", gray, use_sb=True)

# CLAHE contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)
try_detection("CLAHE (classic)", gray_clahe, use_sb=False)
try_detection("CLAHE (SB)", gray_clahe, use_sb=True)

# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
try_detection("Adaptive Threshold (classic)", adaptive, use_sb=False)
try_detection("Adaptive Threshold (SB)", adaptive, use_sb=True)

# Inverted image
inverted = 255 - gray
try_detection("Inverted (classic)", inverted, use_sb=False)
try_detection("Inverted (SB)", inverted, use_sb=True)

# Done
print("\n✅ All methods tested. Check the windows and debug_output/ folder.")
cv2.destroyAllWindows()

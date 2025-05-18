import cv2
import os
import numpy as np

# --- CONFIG ---
folder = "debug_view"
pattern_size = (4, 6)  # inner corners of checkerboard
brightness_threshold = 20
visualize = True

# List all bright frames
images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
print(f"📁 Found {len(images)} images.\n")

# Debug output
os.makedirs("debug_results", exist_ok=True)

for fname in images:
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️  Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    found = False
    reason = ""

    if brightness < brightness_threshold:
        reason = f"Too dark ({brightness:.1f})"
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not found:
            reason = "Checkerboard not detected"
        else:
            cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

    # Annotate and show
    vis = img.copy()
    status = "✅" if found else "❌"
    label = f"{status} {fname} | Brightness: {brightness:.1f}"

    if found:
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    else:
        cv2.putText(vis, reason, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print(f"[{fname}] {status} | {reason or 'OK'}")

    cv2.imwrite(f"debug_results/annotated_{fname}", vis)

    if visualize:
        cv2.imshow("Debug Viewer", vis)
        key = cv2.waitKey(0)
        if key == 27:
            break

cv2.destroyAllWindows()

import cv2
import numpy as np

# Load original checkerboard image (must be grayscale or binary)
checker = cv2.imread("checkerboard.png", cv2.IMREAD_GRAYSCALE)

# Add 100px white border around
bordered = cv2.copyMakeBorder(checker, 100, 100, 100, 100,
                               borderType=cv2.BORDER_CONSTANT,
                               value=255)  # white

cv2.imwrite("checkerboard_bordered.png", bordered)
print("✅ Saved checkerboard_bordered.png with white border")

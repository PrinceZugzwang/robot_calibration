import cv2
import numpy as np

# --- New config ---
squares_x = 4  # 5 inner corners = 6 squares
squares_y = 6  # 7 inner corners = 8 squares
square_size_px = 60  # You can adjust this if needed

width = (squares_x + 1) * square_size_px
height = (squares_y + 1) * square_size_px

# Create checkerboard image
img = np.zeros((height, width), dtype=np.uint8)

for i in range(squares_y + 1):
    for j in range(squares_x + 1):
        if (i + j) % 2 == 0:
            top_left = (j * square_size_px, i * square_size_px)
            bottom_right = ((j + 1) * square_size_px, (i + 1) * square_size_px)
            cv2.rectangle(img, top_left, bottom_right, 255, -1)

cv2.imwrite("checkerboard.png", img)
print("✅ Saved checkerboard.png")

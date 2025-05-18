import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

image_paths = {
    "cam0": "object_images/cam0_object_image.jpg",
    "cam1": "object_images/cam1_object_image.jpg",
    "cam2": "object_images/cam2_object_image.jpg"
}

clicked_points = {}

for cam_name, path in image_paths.items():
    img = mpimg.imread(path)

    fig, ax = plt.subplots()
    ax.set_title(f"Click object center for {cam_name}")
    ax.imshow(img)

    coords = []

    def onclick(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            print(f"{cam_name} clicked at: {x}, {y}")
            plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if coords:
        clicked_points[cam_name] = list(coords[0])

# Save to JSON
with open("object_pixel_coords.json", "w") as f:
    json.dump(clicked_points, f, indent=2)

print("✅ Saved pixel locations to object_pixel_coords.json")

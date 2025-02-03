import cv2
import numpy as np

# X-Trans 6×6 pattern (repeats across the image).
# Each entry is one of 'R', 'G', or 'B'.
xtrans_pattern = [
    ['G','B','R','G','R','B'],
    ['R','G','G','B','G','G'],
    ['B','G','G','R','G','G'],
    ['G','R','B','G','B','R'],
    ['B','G','G','R','G','G'],
    ['R','G','G','B','G','G']
]

def convert_to_xtrans(image_path, output_path):
    # Read the image in BGR form
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not open or find the image.")

    # Height, width, and channels
    h, w, _ = img_bgr.shape

    # Prepare a single-channel array for the mosaic
    # We'll store intensities for the color that belongs to each pixel.
    xtrans_mosaic = np.zeros((h, w), dtype=img_bgr.dtype)

    for y in range(h):
        for x in range(w):
            # Determine which color we should keep at this pixel
            color_filter = xtrans_pattern[y % 6][x % 6]
            if color_filter == 'R':
                xtrans_mosaic[y, x] = img_bgr[y, x, 2]  # Red channel
            elif color_filter == 'G':
                xtrans_mosaic[y, x] = img_bgr[y, x, 1]  # Green channel
            elif color_filter == 'B':
                xtrans_mosaic[y, x] = img_bgr[y, x, 0]  # Blue channel

    # Save the simulated X-Trans mosaic. 
    # This will look odd if viewed directly because it's just the raw pattern.
    cv2.imwrite(output_path, xtrans_mosaic)

if __name__ == "__main__":
    # Example usage
    input_image_path = "orchidBee.jpg"
    output_image_path = "xtrans_mosaic.png"
    convert_to_xtrans(input_image_path, output_image_path)
    print(f"X-Trans mosaic saved to {output_image_path}")
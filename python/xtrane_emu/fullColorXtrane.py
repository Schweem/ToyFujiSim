import cv2
import numpy as np
from demosaic import xtrans_interpolate

# X-Trans 6×6 pattern
xtrans_pattern = [
    ['G','B','R','G','R','B'],
    ['R','G','G','B','G','G'],
    ['B','G','G','R','G','G'],
    ['G','R','B','G','B','R'],
    ['B','G','G','R','G','G'],
    ['R','G','G','B','G','G']
]

def convert_to_xtrans_color(image_path, output_path):
    # Read the image in BGR
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not open or find the image.")

    # Height, width, and channels
    h, w, _ = img_bgr.shape

    # Create a 4-channel mosaic image.
    # The interpolation routine expects a 4-channel image.
    xtrans_mosaic_4ch = np.zeros((h, w, 4), dtype=img_bgr.dtype)

    for y in range(h):
        for x in range(w):
            color_filter = xtrans_pattern[y % 6][x % 6]
            if color_filter == 'R':
                # Assign red value to channel 0.
                xtrans_mosaic_4ch[y, x, 0] = img_bgr[y, x, 2]
            elif color_filter == 'G':
                # Assign green value to channel 1.
                xtrans_mosaic_4ch[y, x, 1] = img_bgr[y, x, 1]
                # Also fill channel 3 (extra channel) with green as default.
                xtrans_mosaic_4ch[y, x, 3] = img_bgr[y, x, 1]
            elif color_filter == 'B':
                # Assign blue value to channel 2.
                xtrans_mosaic_4ch[y, x, 2] = img_bgr[y, x, 0]
                # Optionally, you could assign a default to channel 3 as well.
                # xtrans_mosaic_4ch[y, x, 3] = some_default_value

    # Create the numeric version of the X-Trans pattern.
    xtrans_numeric = np.zeros((6, 6), dtype=np.int32)
    for i in range(6):
        for j in range(6):
            if xtrans_pattern[i][j] == 'G':
                xtrans_numeric[i, j] = 1

    # Perform demosaicing (2 passes, with verbose output).
    xtrans_mosaic_4ch = xtrans_interpolate(xtrans_mosaic_4ch, xtrans_numeric, passes=2, verbose=True)

    # Extract a 3-channel color image from the 4-channel result.
    demosaiced_bgr = cv2.merge([
        xtrans_mosaic_4ch[:, :, 2],  # B
        xtrans_mosaic_4ch[:, :, 1],  # G
        xtrans_mosaic_4ch[:, :, 0]   # R
    ])

    cv2.imwrite(output_path, demosaiced_bgr)
    print(f"Full Color X-Tran demosaic saved to {output_path}")
    

if __name__ == "__main__":
    input_image_path = "orchidBee.jpg"
    output_image_path = "xtrans_mosaic_color.png"
    convert_to_xtrans_color(input_image_path, output_image_path)
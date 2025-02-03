# demoniac.py
import numpy as np

def xtrans_interpolate(xtrans_mosaic_4ch, xtrans_numeric, passes=2, verbose=False):
    """
    A naive multi-pass interpolation for a 4-channel X-Trans mosaic.
    Args:
        xtrans_mosaic_4ch (numpy.ndarray): Image of shape (H, W, 4).
          - Channel 0: Red
          - Channel 1: Green
          - Channel 2: Blue
          - Channel 3: (Extra Green)
        xtrans_numeric (numpy.ndarray): 6x6 array indicating G positions (1) and non-G (0).
        passes (int): How many interpolation passes to run.
        verbose (bool): If True, prints progress messages.
    Returns:
        numpy.ndarray: The interpolated mosaic of the same shape.
    """

    # Convert to float for intermediate calculations
    xtrans_mosaic_4ch = xtrans_mosaic_4ch.astype(np.float32)
    h, w, _ = xtrans_mosaic_4ch.shape

    # We only need to fill channels 0, 1, 2. (Channel 3 is extra green.)
    # For each pass, update missing values (==0) by averaging non-zero neighbors.
    for p in range(passes):
        if verbose:
            print(f"Interpolation Pass {p+1}/{passes}")
        updated = xtrans_mosaic_4ch.copy()

        # Simple 3x3 neighborhood average of non-zero neighbors
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                for c in [0, 1, 2]:
                    if updated[y, x, c] == 0:
                        neighborhood = xtrans_mosaic_4ch[y - 1:y + 2, x - 1:x + 2, c]
                        non_zero_vals = neighborhood[neighborhood > 0]
                        if len(non_zero_vals) > 0:
                            updated[y, x, c] = np.mean(non_zero_vals)
                        else:
                            updated[y, x, c] = 0

        xtrans_mosaic_4ch = updated

    # Convert back to original dtype (likely uint8)
    return xtrans_mosaic_4ch.astype(np.uint8)
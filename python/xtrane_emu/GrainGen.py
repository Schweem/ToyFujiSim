from PIL import Image
import numpy as np
import random

# -- Borrowed from filmgrainer's --
# https://github.com/larspontoppidan/filmgrainer/blob/main/filmgrainer/graingen.py

def _makeGrayNoise(width, height, power):
    buffer = np.zeros([height, width], dtype=int)
    for y in range(height):
        for x in range(width):
            buffer[y, x] = random.gauss(128, power)
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))

def _makeRgbNoise(width, height, power, saturation):
    buffer = np.zeros([height, width, 3], dtype=int)
    intens_power = power * (1.0 - saturation)
    for y in range(height):
        for x in range(width):
            intens = random.gauss(128, intens_power)
            r_noise = random.gauss(0, power) * saturation
            g_noise = random.gauss(0, power) * saturation
            b_noise = random.gauss(0, power) * saturation
            buffer[y, x, 0] = r_noise + intens
            buffer[y, x, 1] = g_noise + intens
            buffer[y, x, 2] = b_noise + intens
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))

def grainGen(width, height, grain_size, power, saturation, seed=1):
    """
    Generates a procedural grain image with optional color saturation.
    - width/height: final output size
    - grain_size: how much the noise is scaled up (1 = 1:1, 2 = half-res, etc.)
    - power: std dev for random.gauss
    - saturation < 0 => grayscale noise
    - saturation >= 0 => color noise
    """
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        img = _makeGrayNoise(noise_width, noise_height, power)
    else:
        img = _makeRgbNoise(noise_width, noise_height, power, saturation)

    # Scale up to final size
    if grain_size != 1.0:
        img = img.resize((width, height), resample=Image.LANCZOS)

    return img
import json
import numpy as np
import cv2
import sys
import tqdm
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from collections import defaultdict
from pathlib import Path

from GrainGen import grainGen
from fullColorXtrane import convert_to_xtrans_color

class FujiFilmEmulator:
    """
    FujiFilmEmulator attempts to simulate the look of Fuji's film simulations
    using a set of JSON recipes. The recipes are loaded from a directory of
    JSON files, each containing a set of parameters for a specific film
    simulation.
    
    Ingests a picture which has been converted to xtran fromat and 
    demoasiced using Frank Markesteijn's demosaicing algorithm. 
    
    The emulator supports the following parameters:
    - DynamicRange: Simulates Fuji's DR modes by compressing highlights.
    - Highlight: Adjusts the tone curve to compress highlights.
    - Shadow: Adjusts the tone curve to lift shadows.
    - Color: Adjusts the color profile to enhance saturation.
    - Clarity: Enhances midtone contrast.
    - GrainEffect: Adds a grain pattern to the image.
    - GrainSize: Controls the size of the grain pattern.
    - ColorChromeEffect: Boosts saturation selectively.
    - ColorChromeEffectBlue: Boosts blue saturation selectively.
    - WhiteBalanceShift: Adjusts the white balance by shifting RGB channels.
    - ExposureCompensation: Adjusts the overall brightness of the image.
    
    The emulator also includes a set of base curves for Fuji's film simulations.
    
    """
    def __init__(self, recipes_dir='fuji_recipes', curves_dir='base_curves'):
        self.recipes = self._load_recipes(recipes_dir)
        self.base_curves = self._load_base_curves(curves_dir) 
        
        
    def _create_base_curve(self, strength=1.0, highlight_rolloff=1.0):
        """ Create a base curve LUT for highlight compression """
        
        x = np.linspace(0, 1, 256)
        
        # Use the 'strength' parameter directly as the gamma value.
        gamma_curve = x ** strength

        threshold = 0.7  # Adjust this threshold as needed.
        mask = gamma_curve > threshold
        compressed = gamma_curve.copy()
        # Compress highlights above the threshold.
        compressed[mask] = threshold + (compressed[mask] - threshold) * highlight_rolloff

        curve = np.clip(compressed * 255, 0, 255).astype(np.uint8)
        return curve

    def apply_base_curve(self, img, curve_name):
        """ Apply a base curve to an image """
        
        # Get the loaded curve data from JSON
        curve_data = self.base_curves[curve_name]

        # If curve_data is a dict with parameters, generate the LUT
        if isinstance(curve_data, dict):
            strength = curve_data.get("strength", 1.0)
            highlight_rolloff = curve_data.get("highlight_rolloff", 1.0)
            lut = self._create_base_curve(strength, highlight_rolloff)
        else:
            # Otherwise assume it's already a LUT list
            lut = np.array(curve_data, dtype=np.uint8)

        arr = np.array(img)
        # Apply the lookup table to each color channel
        for c in range(3):
            arr[:, :, c] = cv2.LUT(arr[:, :, c], lut)
        return Image.fromarray(arr)
    
    
    def apply_recipe(self, image_path, recipe_name, base_curve):
        """ Apply a recipe to an image """
        
        img = Image.open(image_path).convert('RGB')
        recipe = self.recipes[recipe_name]
        
        # Apply processing pipeline
        img = self.apply_base_curve(img, base_curve)
        img = self._apply_dynamic_range(img, recipe['DynamicRange'])
        img = self._apply_tone_curve(img, recipe['Highlight'], recipe['Shadow'])
        img = self._apply_color_profile(img, recipe['Color'])
        img = self._apply_clarity(img, recipe['Clarity'])
        img = self._apply_grain(img, recipe['GrainEffect'], recipe['GrainSize'])
        img = self._apply_color_chrome(img, recipe['ColorChromeEffect'], recipe['ColorChromeEffectBlue'])
        img = self._apply_white_balance(img, recipe['WhiteBalanceShift'])
        img = self._apply_exposure_compensation(img, recipe['ExposureCompensation'])
        
        return img

    def _load_recipes(self, directory):
        """ Load all JSON recipes from a directory """
        
        # Load all JSON recipes from directory
        recipes = {}
        for recipe_file in Path(directory).glob('*.json'):
            with open(recipe_file, 'r') as f:
                recipe = json.load(f)
                recipes[recipe_file.stem] = recipe
        
        
        return recipes
    
    def _load_base_curves(self, directory):
        """ Load all JSON base curves from a directory """
        
        # Load all JSON base curves from directory
        base_curves = {}
        for curve_file in Path(directory).glob('*.json'):
            with open(curve_file, 'r') as f:
                curve = json.load(f)
                base_curves[curve_file.stem] = curve
        
        return base_curves

    def _apply_dynamic_range(self, img, dr_value):
        """
        Simulate Fuji DR modes by selectively compressing the highlights.
        DR100 => no change
        DR200 => moderate highlight compression
        DR400 => stronger highlight compression
        """
        
        # Map DR values to highlight compression factors
        compression_map = {
            'DR100': 1.0,  # no change
            'DR200': 0.85, # moderate
            'DR400': 0.70  # stronger
        }
        highlight_compression = compression_map.get(dr_value, 1.0)

        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Define a threshold above which we compress highlights
        threshold = 0.7  # everything above 70% brightness gets compressed
        mask_high = arr > threshold

        # For the highlight region, map from [threshold..1] down to [threshold..(something lower)]
        # We'll do a linear compression for that portion
        arr_high = arr[mask_high]
        # Move it to [0..(1-threshold)] range first
        arr_high -= threshold

        # Apply highlight compression factor
        arr_high *= highlight_compression

        # Shift it back up
        arr_high += threshold

        # Clip and put it back
        arr[mask_high] = np.clip(arr_high, 0, 1.0)

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(arr)
    
    def _apply_clarity(self, img, clarity_strength):
        """Simulates Fuji's clarity effect by enhancing midtone contrast."""
        
        if clarity_strength == 0:
            return img  # No clarity adjustment needed

        # Convert to grayscale for edge detection
        gray = img.convert("L")

        # Apply a high-pass filter to extract midtone contrast
        high_pass = gray.filter(ImageFilter.FIND_EDGES)

        # Normalize high-pass filter and adjust strength
        hp_array = np.array(high_pass).astype(np.float32)
        hp_array = (hp_array - hp_array.min()) / (hp_array.max() - hp_array.min() + 1e-6) * 255
        hp_array = np.clip(hp_array * (clarity_strength / 10), 0, 255).astype(np.uint8)

        # Convert back to an image
        hp_img = Image.fromarray(hp_array)

        # Blend high-pass filter with the original image
        return Image.blend(img, hp_img.convert("RGB"), clarity_strength / 20)

    def _apply_clarity_cv(self, pil_img, clarity_strength=1.2):
        """
        Applies a mid-frequency unsharp mask in the Lab color space to enhance 'clarity'.
        
        Args:
            pil_img (PIL.Image): The source image in RGB.
            clarity_strength (float): Multiplicative factor for midtone detail.
                - 1.0 = no change,
                - >1.0 = stronger clarity,
                - <1.0 = reduces clarity (softens the image).
        
        Returns:
            PIL.Image: The processed image in RGB.
        """
        
        # Convert PIL image to OpenCV (BGR) format
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert BGR to Lab
        lab_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_img)

        # Convert L channel to float for processing
        L_f = L.astype(np.float32)

        # Apply a relatively large Gaussian blur to extract low-frequency content
        blur_radius = 13  # tweak radius to taste; larger radius = more 'broad' clarity effect
        blurred_L = cv2.GaussianBlur(L_f, (0, 0), blur_radius)

        # The difference between the original L and blurred_L is mid-frequency detail
        detail = L_f - blurred_L

        # Scale the detail by clarity_strength (above 1 => stronger detail, below 1 => softer)
        L_new = L_f + clarity_strength * detail

        # Clip to valid range and convert back to uint8
        L_new = np.clip(L_new, 0, 255).astype(np.uint8)

        # Merge channels back
        lab_new = cv2.merge([L_new, a, b])
        # Convert back to BGR
        bgr_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
        # Finally convert to RGB for PIL
        rgb_new = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2RGB)

        return Image.fromarray(rgb_new)

    def _apply_tone_curve(self, img, highlight, shadow):
        """
        Refined tone curve with separate shadow and highlight control.
        highlight > 0 => Compress highlights
        shadow > 0   => Lift shadows
        """

        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Convert highlight & shadow to gamma factors
        # e.g., highlight=50 means we compress the top end more strongly
        gamma_shadow = 1.0 / (1.0 + shadow / 100.0)   # >1 => lifts shadows
        gamma_high = 1.0 / (1.0 + highlight / 100.0) # >1 => compresses highlights

        # Midpoint for piecewise
        mid = 0.5

        # Piecewise function
        # Shadow region [0..mid]
        mask_shadow = arr < mid
        arr_shadow = (arr[mask_shadow] / mid) ** gamma_shadow
        arr[mask_shadow] = arr_shadow * mid

        # Highlight region [mid..1]
        mask_high = arr >= mid
        arr_high = (arr[mask_high] - mid) / (1.0 - mid)
        arr_high = arr_high ** gamma_high
        arr[mask_high] = mid + arr_high * (1.0 - mid)

        # Scale back to [0..255]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(arr)

    def _apply_color_profile(self, img, color_strength):
        """Fuji color rendering with enhanced saturation and specific hue shifts"""
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1 + color_strength/100)
        
        # Fuji-specific color matrix adjustment
        arr = np.array(img)
        # Boost reds and greens slightly
        arr[:,:,0] = np.clip(arr[:,:,0] * 1.05, 0, 255)  # Red channel
        arr[:,:,1] = np.clip(arr[:,:,1] * 1.03, 0, 255)  # Green channel
        
        return Image.fromarray(arr.astype(np.uint8))

    def _apply_grain(self, img, strength, size):
        """
        Procedural grain using filmgrainer-like approach.
        'strength' can map to (power, saturation, etc.).
        'size' can map to the grain_size param or be a separate concept (Small vs Large).
        """
        
        grain_strength_map = {
            "Off": (0, 0.0),
            "Weak": (8, 0.2),   # (power, saturation) example
            "Strong": (16, 0.4)
        }
        power, saturation = grain_strength_map.get(strength, (8, 0.2))
        if power == 0:
            return img  # "Off" => no grain

        # Decide how big you want each 'grain pixel' to be
        # e.g. size = 'Small' => smaller grain_size factor
        if size == 'Small':
            grain_size_factor = 2.0
        else:
            grain_size_factor = 1.0

        # Generate the grain
        width, height = img.size
        grain_img = grainGen(width, height, grain_size_factor, power, saturation)

        # Blend: alpha can be controlled by you or baked into the "strength" mapping
        alpha = 0.15 if strength == "Weak" else 0.25  # tweak to taste

        # Convert to arrays and combine
        base_arr = np.array(img, dtype=np.float32)
        grain_arr = np.array(grain_img, dtype=np.float32)

        # If the grain image is grayscale, broadcast it to 3 channels
        if len(grain_arr.shape) == 2:
            grain_arr = np.stack((grain_arr,)*3, axis=-1)

        # Basic blend
        out_arr = base_arr * (1 - alpha) + grain_arr * alpha
        out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)

        return Image.fromarray(out_arr)

    def _apply_color_chrome(self, pil_img, strength, blue_strength):
        """
        A more 'selective saturation boost' approach in Lab.
        strength, blue_strength could be 0..2 for example, where 1.0 means no change.
        """
        
        # Map textual presets to numeric
        chrome_map = {
            "Weak": 1.1,
            "Strong": 1.3,
            "Off": 1.0
        }
        # Default to 1.2 if not recognized
        color_chrome_factor = chrome_map.get(strength, 1.2)
        blue_factor = chrome_map.get(blue_strength, 1.2)

        # Convert PIL image to OpenCV
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert BGR to Lab
        lab_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)

        # Split channels
        L, a, b = cv2.split(lab_img)

        # Convert to float
        a_f = a.astype(np.float32) - 128.0
        b_f = b.astype(np.float32) - 128.0

        # Compute saturation in Lab
        saturation = np.sqrt(a_f**2 + b_f**2)

        # We can define a threshold above which we boost saturation more aggressively
        # e.g., 40 is a moderate Lab saturation threshold
        threshold = 40.0
        mask = (saturation > threshold)

        # For saturated areas, boost a_f and b_f
        a_f[mask] *= color_chrome_factor
        b_f[mask] *= color_chrome_factor

        # If we want to specifically modify blues, we can do a selective approach:
        # Identify hue roughly in Lab by checking if a_f is negative and b_f is negative 
        # (blue/cyan region). This is a naive approach
        blue_mask = (a_f < 0) & (b_f < 0)
        a_f[blue_mask] *= blue_factor
        b_f[blue_mask] *= blue_factor

        # Re-add 128 for proper Lab range, clip to [0..255]
        a_new = np.clip(a_f + 128.0, 0, 255).astype(np.uint8)
        b_new = np.clip(b_f + 128.0, 0, 255).astype(np.uint8)

        # Reassemble lab
        lab_new = cv2.merge([L, a_new, b_new])

        # Convert back to BGR
        bgr_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

        # Convert BGR back to PIL (RGB)
        rgb_new = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(rgb_new)

        return result_img

    def _apply_white_balance(self, img, shifts):
        """Apply RGB channel shifts with Fuji's auto balance characteristics"""
        
        r_gain = 1 + shifts['Red']/100
        b_gain = 1 + shifts['Blue']/100
        arr = np.array(img).astype(np.float32)
        arr[:,:,0] *= r_gain  # Red channel
        arr[:,:,2] *= b_gain  # Blue channel
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _apply_exposure_compensation(self, img, comp):
        """Fuji-style exposure compensation with highlight protection"""
        
        return ImageEnhance.Brightness(img).enhance(1 + comp/3)

def main():
    """ Command-line interface for the FujiFilmEmulator """
    
    if sys.argv[1] == '--help':
        print('\nUsage: python FujiFilmEmulator.py <photo> <preset> <curve>')
        print('\nExample: python FujiFilmEmulator.py photo.jpg classic_chrome provia')
        print('\nUse --list to see available presets and base curves.')
        return
    
    if sys.argv[1] == '--list':
        print('\nAvailable presets:\n----------')
        for preset in Path('fuji_presets').glob('*.json'):
            print(preset.stem)
        print('\nAvailable base curves:\n----------')
        for curve in ['classic_chrome', 'provia', 'astia', 'velvia']: #TODO un-hardcode this
            print(curve)
            
        return
    
    if len(sys.argv) < 4:
        print('Usage: python FujiFilmEmulator.py <photo> <preset> <curve>')
        return
    
    photo = sys.argv[1]
    filename = photo.split('.')[0]
    preset = sys.argv[2]
    curve = sys.argv[3]
    
    emulator = FujiFilmEmulator(recipes_dir='fuji_presets', curves_dir='base_curves')
    
    tqdm.tqdm.write(f'Converting {photo} to xtrans format.')
    xtran_image = convert_to_xtrans_color(photo, f'./outputs/{filename}_xtrans.jpg')
    tqdm.tqdm.write('Success! \n')
    
    tqdm.tqdm.write(f'Applying preset {preset} with {curve} base curve to {filename}')
    tqdm.tqdm.write(f'Output will be saved to ./outputs/{curve}_{preset}_{filename}.jpg')
    
    result = emulator.apply_recipe(f'./outputs/{filename}_xtrans.jpg', preset, curve)
    
    result.save(f'./outputs/{curve}_{preset}_{filename}.jpg')
    
if __name__ == "__main__":
    
    main()
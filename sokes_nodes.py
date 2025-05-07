from datetime import datetime # datetime
import re # regex
import os # for load random image
import random # for load random image
import hashlib # for random number
import imghdr # to check image by header vs extension
import torch # to resize images
import numpy as np # for image manipulation in torch
import cv2 # for image processing
from PIL import Image, ImageOps # load random image, orient image
 
# Might need to: pip install webcolors colormath
import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000
 
from .sokes_color_maps import css3_names_to_hex, css3_hex_to_names, human_readable_map, explicit_targets_for_comparison
 
import numpy as np
if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: a.item()        
 
 
##############################################################
# START Current Date | Sokes 🦬
 
class current_date_sokes:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "date_format": ("STRING", {"default": 'YYYY-MM-DD', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_date",)
    FUNCTION = "current_date_sokes"
    CATEGORY = "Sokes 🦬"
    
    def current_date_sokes(self, date_format):
        now = datetime.now()  # Fresh timestamp on every execution
    
        # Uppercase date/time components (case-insensitive)
        formatted = re.sub(r'(?i)(y+|m+|d+)', lambda m: m.group().upper(), date_format)
 
        # Replace in order: longest first to prevent partial replacements
        # Year
        formatted = formatted.replace("YYYY", now.strftime("%Y"))
        formatted = formatted.replace("YY", now.strftime("%y"))
        
        # Month 
        formatted = formatted.replace("MM", now.strftime("%m"))  # Zero-padded
        formatted = formatted.replace("M", now.strftime("%m").lstrip("0"))  # No zero-pad
        
        # Day
        formatted = formatted.replace("DD", now.strftime("%d"))  # Zero-padded
        formatted = formatted.replace("D", now.strftime("%d").lstrip("0"))  # No zero-pad
 
        return (formatted,)
 
    @classmethod
    def IS_CHANGED(cls, date_format):
        # Force re-execution
        return (datetime.now().timestamp(),)
 
# END Current Date | Sokes 🦬
##############################################################
 
 
##############################################################
# START Latent Input Swtich x9 | Sokes 🦬
 
class latent_input_switch_9x_sokes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_select": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "latent_0": ("LATENT",),
            },
            "optional": {
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
                "latent_6": ("LATENT",),
                "latent_7": ("LATENT",),
                "latent_8": ("LATENT",),
            },
        }
 
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "latent_input_switch_9x_sokes"
    OUTPUT_NODE = True
    CATEGORY = "Sokes 🦬"
 
    def latent_input_switch_9x_sokes(self, latent_select, latent_0, latent_1=None, latent_2=None, latent_3=None, latent_4=None, latent_5=None, latent_6=None, latent_7=None, latent_8=None):
        if int(round(latent_select)) == 0 and latent_0 != None:
            return (latent_0, )
        if int(round(latent_select)) == 1 and latent_1 != None:
            return (latent_1, )
        if int(round(latent_select)) == 2 and latent_2 != None:
            return (latent_2, )
        if int(round(latent_select)) == 3 and latent_3 != None:
            return (latent_3, )
        if int(round(latent_select)) == 4 and latent_4 != None:
            return (latent_4, )
        if int(round(latent_select)) == 5 and latent_5 != None:
            return (latent_5, )
        if int(round(latent_select)) == 6 and latent_6 != None:
            return (latent_6, )
        if int(round(latent_select)) == 7 and latent_7 != None:
            return (latent_7, )
        if int(round(latent_select)) == 8 and latent_8 != None:
            return (latent_8, )
        else:
            return (latent_0, )
 
# END Latent Input Swtich x9 | Sokes 🦬
##############################################################
 
##############################################################
# START Replace Text with Regex | Sokes 🦬
 
class replace_text_regex_sokes:
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "regex_pattern": ("STRING", {"multiline": False}),
            "new": ("STRING", {"multiline": False})
        }}
 
    RETURN_TYPES = ("STRING",)
    FUNCTION = "fn_replace_text_regex_sokes"
    CATEGORY = "Sokes 🦬"
 
    @ staticmethod
    def fn_replace_text_regex_sokes(regex_pattern, new, text):
        return (re.sub(regex_pattern, new, text),)
 
 
# END Replace Text with Regex | Sokes 🦬
##############################################################
 
##############################################################
# START: Load Random Image | Sokes 🦬

class load_random_image_sokes:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": "."}),
                "n_images": ("INT", {"default": 1, "min": -1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "sort": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Sokes 🦬"
    RETURN_TYPES = ("IMAGE", "LIST")
    RETURN_NAMES = ("image", "image_path")
    FUNCTION = "load_image"

    def load_image(self, folder, n_images, seed, sort):
        # Get all files in the folder
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        image_paths = [f for f in image_paths if os.path.isfile(f)]
        image_paths = [f for f in image_paths if any([f.endswith(ext) for ext in self.img_extensions])]

        # Validate images
        valid_image_paths = []
        for f in image_paths:
            if imghdr.what(f):
                valid_image_paths.append(f)
            else:
                try:
                    img = Image.open(f)
                    img.verify()  # Verify that the file is a valid image
                    valid_image_paths.append(f)
                except Exception as e:
                    print(f"Skipping invalid image: {f} - {str(e)}")

        # Check if no valid images were found
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")

        # Shuffle or sort the images
        random.seed(seed)
        random.shuffle(valid_image_paths)

        if n_images > 0:
            valid_image_paths = valid_image_paths[:n_images]

        if sort:
            valid_image_paths = sorted(valid_image_paths)

        # Load and process images
        imgs = []
        for image_path in valid_image_paths:
            try:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(f"Error during EXIF transpose for {image_path}: {str(e)}")
                continue  # Skip this image on error

            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))

            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            imgs.append(image)

        if not imgs:
            raise ValueError("No images were successfully loaded.")

        # Process images into a single tensor
        if len(imgs) > 1:
            imgs = create_same_sized_crops(imgs, target_n_pixels=1024**2)
            imgs = [torch.from_numpy(img)[None,] for img in imgs]
            output_image = torch.cat(imgs, dim=0)
        else:
            output_image = torch.from_numpy(imgs[0])[None,]

        #filenames_str = ", ".join(valid_image_paths)  # Comma-separated

        return (output_image, valid_image_paths)

# END: Load Random Image | Sokes 🦬
##############################################################
 
##############################################################
# START Hex to Color Name | Sokes 🦬
 
class hex_to_color_name_sokes:
    CATEGORY = "Sokes 🦬"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("color_name",)
    FUNCTION = "hex_to_color_name_fn"
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": (
                    "STRING",
                    {"default": "#FFFFFF", "multiline": False},
                )
            },
            "optional": {
                "use_css_name": ("BOOLEAN", {"default": False})
            }
        }
 
    def hex_to_color_name_fn(self, hex_color, use_css_name=False):
        # ... Input Sanitization ...
        if not hex_color: return ("Input hex color is empty.",)
        hex_color = hex_color.strip()
        if not hex_color.startswith("#"): hex_color = "#" + hex_color
        if len(hex_color) == 4: hex_color = f"#{hex_color[1]*2}{hex_color[2]*2}{hex_color[3]*2}"
        if len(hex_color) != 7: return (f"Invalid hex format: {hex_color}",)
 
 
        # --- 1) Exact CSS3 Match ---
        try:
            standard_name = webcolors.hex_to_name(hex_color, spec="css3")
            final_color_name = standard_name
            if not use_css_name:
                 final_color_name = human_readable_map.get(standard_name, standard_name)
                 if final_color_name is None: final_color_name = standard_name
            return (final_color_name,)
        except ValueError:
            # --- 2) Fallback: Find Nearest Color ---
            try:
                requested_rgb = webcolors.hex_to_rgb(hex_color)
                requested_lab = convert_color(sRGBColor(*requested_rgb, is_upscaled=True), LabColor)
            except Exception as e: return (f"Invalid input hex/conversion error: {e}",)
 
            min_dist = float('inf')
            closest_name = None
            checked_hex_codes = set()
 
            # --- Check EXPLICIT TARGETS first ---
            for target_name, target_hex in explicit_targets_for_comparison.items():
                 if len(target_hex) == 4: target_hex = f"#{target_hex[1]*2}{target_hex[2]*2}{target_hex[3]*2}"
                 if target_hex in checked_hex_codes: continue
                 try:
                     target_rgb = webcolors.hex_to_rgb(target_hex)
                     target_lab = convert_color(sRGBColor(*target_rgb, is_upscaled=True), LabColor)
                     d = delta_e_cie2000(requested_lab, target_lab)
                     if d < min_dist:
                         min_dist = d
                         closest_name = target_name
                     checked_hex_codes.add(target_hex)
                 except Exception as e: print(f"Warn: Cannot process explicit target {target_name}: {e}")
 
 
            # --- Iterate through STANDARD CSS3 colors ---
            # <<< Use the imported css3_names_to_hex dictionary here >>>
            for name, candidate_hex in css3_names_to_hex.items():
                 if len(candidate_hex) == 4: candidate_hex = f"#{candidate_hex[1]*2}{candidate_hex[2]*2}{candidate_hex[3]*2}"
                 if candidate_hex in checked_hex_codes: continue # Skip if already checked
                 try:
                    cand_rgb = webcolors.hex_to_rgb(candidate_hex)
                    cand_lab = convert_color(sRGBColor(*cand_rgb, is_upscaled=True), LabColor)
                    d = delta_e_cie2000(requested_lab, cand_lab)
                    if d < min_dist:
                        min_dist = d
                        closest_name = name # Update to the standard CSS3 name
                    checked_hex_codes.add(candidate_hex) # Add here to avoid potential re-check errors
                 except Exception as e: print(f"Warn: Error processing candidate {name}: {e}")
 
            if closest_name is None: return ("Could not find closest color.",)
 
            # --- 3) Map the closest internal name found ---
            final_color_name = closest_name
            if not use_css_name:
                mapped_name = human_readable_map.get(closest_name)
                if mapped_name: final_color_name = mapped_name
                else: final_color_name = closest_name # Fallback if no mapping
 
            return (final_color_name,)
 
# END: Hex to Color Name | Sokes 🦬
##############################################################

##############################################################
# START Random Number | Sokes 🦬
# Fixed bug and now maintaining. Originally from WAS_Node_Suite / WAS_Random_Number
class random_number_sokes:
    """
    Generates a primary random float, and derives an integer (rounded)
    and a boolean from it. All outputs are always populated based on
    a single random float generation.
    """
    CATEGORY = "Sokes 🦬"

    # This defines the order and type of outputs from left to right on the node
    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("integer_output", "float_output", "boolean_output")
    FUNCTION = "generate_random_value"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "minimum": ("FLOAT", {"default": 0.0, "min": -1.0e18, "max": 1.0e18, "step": 0.01}),
                "maximum": ("FLOAT", {"default": 1.0, "min": -1.0e18, "max": 1.0e18, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def generate_random_value(self, minimum, maximum, seed):
        random.seed(seed)

        # 1. Generate the primary random float
        primary_float = random.uniform(minimum, maximum)

        # 2. Derive the integer output
        derived_int = int(round(primary_float))

        # 3. Derive the boolean output
        midpoint = minimum + (maximum - minimum) / 2.0
        derived_bool = primary_float > midpoint

        # Return values in the order defined by RETURN_TYPES and RETURN_NAMES
        # INT first, then FLOAT, then BOOLEAN
        return (derived_int, primary_float, derived_bool)

    @classmethod
    def IS_CHANGED(cls, minimum, maximum, seed):
        m = hashlib.sha256()
        m.update(str(minimum).encode('utf-8'))
        m.update(str(maximum).encode('utf-8'))
        m.update(str(seed).encode('utf-8'))
        return m.digest().hex()
# END: Random Number | Sokes 🦬
##############################################################


##############################################################
# Node Mappings
 
NODE_CLASS_MAPPINGS = {
    "Current Date | sokes 🦬": current_date_sokes,
    "Latent Switch x9 | sokes 🦬": latent_input_switch_9x_sokes,
    "Replace Text with RegEx | sokes 🦬": replace_text_regex_sokes,
    "Load Random Image | sokes 🦬": load_random_image_sokes,
    "Hex to Color Name | sokes 🦬": hex_to_color_name_sokes,
    "Random Number | sokes 🦬": random_number_sokes,
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "Current Date | sokes 🦬": "Current Date 🦬",
    "Latent Switch x9 | sokes 🦬": "Latent Switch x9 🦬",
    "Replace Text with RegEx | sokes 🦬": "Replace Text with RegEx 🦬",
    "Load Random Image | sokes 🦬": "Load Random Image 🦬",
    "Hex to Color Name | sokes 🦬": "Hex to Color Name 🦬",
    "Random Number | sokes 🦬": "Random Number 🦬",
}
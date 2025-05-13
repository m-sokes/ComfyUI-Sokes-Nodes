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
# START Load Random Image with Path and Mask | Sokes 🦬 (with subfolder search)

class load_random_image_sokes:
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "."}),
                "include_subfolders": ("BOOLEAN", {"default": False}), # <<< New Toggle
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sort": ("BOOLEAN", {"default": False}),
                "export_with_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image"

    def find_image_files(self, folder_path, include_subfolders):
        """Helper function to find image files, optionally searching subfolders."""
        image_paths = []
        img_extensions_lower = [ext.lower() for ext in self.IMG_EXTENSIONS]
        if include_subfolders:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(tuple(img_extensions_lower)):
                        full_path = os.path.join(root, file)
                        image_paths.append(full_path)
        else:
            try:
                all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(tuple(img_extensions_lower))]
            except Exception as e:
                raise Exception(f"Error listing files in folder '{folder_path}': {e}")
        return image_paths

    def load_image(self, folder, include_subfolders, n_images, seed, sort, export_with_alpha): # Added include_subfolders
        resolved_folder_path = folder
        if not os.path.isdir(folder):
            # Try resolving using ComfyUI's folder_paths if available
            if folder_paths:
                try:
                    input_dir = folder_paths.get_input_directory()
                    annotated_path = folder_paths.get_annotated_filepath(folder)
                    if annotated_path and os.path.isdir(annotated_path):
                        resolved_folder_path = annotated_path
                    elif input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                        resolved_folder_path = os.path.join(input_dir, folder)
                    #else: pass # Keep original 'folder' if not resolved and no exception
                except NameError: # Should not happen if folder_paths is not None, but defensive
                     print(f"sokes_nodes.py: 'folder_paths' module check failed unexpectedly for resolving '{folder}'. Using path as is.")
                except Exception as e:
                    print(f"sokes_nodes.py: Error trying to resolve folder '{folder}' using folder_paths: {e}. Using path as is.")
            else:
                print(f"sokes_nodes.py: 'folder_paths' not available. Cannot resolve relative path '{folder}' against input/annotated directories.")

        if not os.path.isdir(resolved_folder_path):
            raise FileNotFoundError(f"Folder '{resolved_folder_path}' (resolved from '{folder}') not found or not a directory.")

        # --- Find Image Files (using new helper function) ---
        try:
            image_paths_found = self.find_image_files(resolved_folder_path, include_subfolders)
        except Exception as e:
             # find_image_files already raises specific exceptions, but catch others
            raise Exception(f"Error finding image files in '{resolved_folder_path}' (subfolders: {include_subfolders}): {e}")


        # --- Validate Images ---
        valid_image_paths = []
        for f_path in image_paths_found:
            try:
                # Basic check first
                is_image_basic = any([f_path.lower().endswith(ext) for ext in self.IMG_EXTENSIONS])
                if not is_image_basic: continue

                # More robust check (imghdr, PIL verification)
                is_image_hdr = imghdr.what(f_path)
                if is_image_hdr: # imghdr recognizes it
                     valid_image_paths.append(f_path)
                     continue
                else: # Try PIL verification as fallback
                    try:
                        img_test = Image.open(f_path)
                        img_test.verify() # Check for corruption
                        valid_image_paths.append(f_path) # If verify() doesn't raise, it's likely a valid image format PIL understands
                    except Exception as pil_e:
                        print(f"Skipping file (PIL verify failed): {f_path} - {str(pil_e)}")

            except Exception as e:
                 print(f"Skipping potentially corrupt or unreadable image: {f_path} - {str(e)}")


        if not valid_image_paths:
            search_scope = "and its subfolders" if include_subfolders else " (subfolders not searched)"
            raise ValueError(f"No valid images found in folder: {resolved_folder_path} {search_scope}")

        # --- Sort or Shuffle ---
        if not sort:
            random.seed(seed)
            random.shuffle(valid_image_paths)
        else:
             valid_image_paths = sorted(valid_image_paths)

        # --- Select Images ---
        num_available = len(valid_image_paths)
        actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
        if actual_n_images < n_images and n_images > 0 :
             print(f"Warning: Requested {n_images} images, but only {num_available} were found/valid in '{resolved_folder_path}'. Loading {actual_n_images}.")

        selected_paths = valid_image_paths[:actual_n_images]

        # --- Load and Process Images ---
        output_images = []
        output_masks = []
        loaded_paths = []
        first_image_shape_hw = None # For H, W consistency check

        # Determine batch output mode if export_with_alpha is True
        force_batch_to_rgba = False
        if export_with_alpha and selected_paths:
            for image_path_check in selected_paths:
                try:
                    # Optimization: Peek at image info without fully loading if possible
                    with Image.open(image_path_check) as img_pil_check:
                         if img_pil_check.mode == 'RGBA' or img_pil_check.mode == 'LA' or (img_pil_check.mode == 'P' and 'transparency' in img_pil_check.info):
                            force_batch_to_rgba = True
                            break # Found one with alpha, entire batch will be RGBA
                except Exception as e:
                    print(f"⚠️ Warning: Could not pre-check image {image_path_check} for alpha: {e}. Assuming no alpha for this check.")

        final_image_mode = "RGB" # Default
        if export_with_alpha and force_batch_to_rgba:
            final_image_mode = "RGBA"
        elif not export_with_alpha: # Default behavior
            final_image_mode = "RGB"
        # If export_with_alpha is True but no image in batch has alpha, final_image_mode remains "RGB"


        for image_path in selected_paths:
            try:
                img_pil = Image.open(image_path)
                img_pil = ImageOps.exif_transpose(img_pil) # Handle EXIF orientation

                # Convert image for IMAGE output based on determined mode
                if img_pil.mode == final_image_mode:
                     converted_image_pil = img_pil
                else:
                    converted_image_pil = img_pil.convert(final_image_mode)

                image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # Shape will be [1, H, W, 3] or [1, H, W, 4]

                # Mask Processing (always check original PIL image's alpha)
                mask_tensor = None
                if img_pil.mode == 'RGBA' or img_pil.mode == 'LA':
                    # Directly use the alpha channel
                    mask_pil_channel = img_pil.split()[-1]
                    mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np)[None,]
                elif img_pil.mode == 'P' and 'transparency' in img_pil.info:
                    # Convert P mode with transparency to RGBA to extract mask
                    img_rgba_for_mask = img_pil.convert('RGBA')
                    mask_pil_channel = img_rgba_for_mask.split()[-1]
                    mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np)[None,]
                else:
                    # No alpha channel, create a full white mask
                    mask_shape = (image_tensor.shape[1], image_tensor.shape[2]) # Use H, W from image tensor
                    mask_np = np.ones(mask_shape, dtype=np.float32)
                    mask_tensor = torch.from_numpy(mask_np)[None,]


                # Batch Consistency Check (H, W dimensions)
                current_shape_hw = image_tensor.shape[1:3] # H, W
                if first_image_shape_hw is None:
                    first_image_shape_hw = current_shape_hw
                elif current_shape_hw != first_image_shape_hw:
                    print(f"⚠️ Warning: Image {os.path.basename(image_path)} has dimensions ({current_shape_hw}) "
                          f"different from the first image ({first_image_shape_hw}). "
                          "Batching may fail or produce unexpected results downstream if dimensions are not consistent. "
                          "Consider resizing images beforehand or using a node that handles varied batch item sizes.")
                    # If you wanted to force resize (example, not fully implemented here):
                    # from torchvision.transforms.functional import resize
                    # target_h, target_w = first_image_shape_hw
                    # image_tensor = resize(image_tensor.permute(0,3,1,2), [target_h, target_w]).permute(0,2,3,1)
                    # mask_tensor = resize(mask_tensor.unsqueeze(1), [target_h, target_w]).squeeze(1)

                output_images.append(image_tensor)
                output_masks.append(mask_tensor)
                loaded_paths.append(image_path)

            except Exception as e:
                print(f"❌ Error loading or processing image {image_path}: {str(e)}. Skipping.")
                continue # Skip this image and proceed to the next

        if not output_images:
             # This can happen if all selected images failed during loading/processing
            raise ValueError("No images were successfully loaded after processing, though valid files were initially found.")

        # Concatenate the lists of tensors into batches
        # torch.cat should now work as channels are consistent across the batch for output_images
        final_image_batch = torch.cat(output_images, dim=0)
        final_mask_batch = torch.cat(output_masks, dim=0)

        return (final_image_batch, final_mask_batch, loaded_paths)

    @classmethod
    def IS_CHANGED(cls, folder, include_subfolders, n_images, seed, sort, export_with_alpha): # Added include_subfolders
        resolved_folder_path = folder
        if not os.path.isdir(folder):
             # Try resolving using ComfyUI's folder_paths if available
            if folder_paths:
                try:
                    annotated_path = folder_paths.get_annotated_filepath(folder)
                    if annotated_path and os.path.isdir(annotated_path):
                        resolved_folder_path = annotated_path
                    else:
                        input_dir = folder_paths.get_input_directory()
                        if input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                            resolved_folder_path = os.path.join(input_dir, folder)
                except NameError: pass # Ignore if folder_paths not fully functional
                except Exception as e:
                    print(f"sokes_nodes.py IS_CHANGED: Error during folder_path resolution: {e}")
                    pass # Fallback to using 'folder' as is if resolution fails
            #else: pass # Keep original 'folder' if folder_paths not available

        if not os.path.isdir(resolved_folder_path):
            # If the folder doesn't exist, treat it as changed (or trigger an error upstream)
             # Using a unique string including the non-existent path helps differentiate
            return f"folder_not_found_{resolved_folder_path}_{include_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"

        # --- Find Image Files (Mirroring the logic in load_image) ---
        image_paths = []
        img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS]
        try:
            if include_subfolders:
                 for root, _, files in os.walk(resolved_folder_path):
                    for file in files:
                        if file.lower().endswith(tuple(img_extensions_lower)):
                             full_path = os.path.join(root, file)
                             # Check if it's actually a file (os.walk might list other things)
                             if os.path.isfile(full_path):
                                image_paths.append(full_path)
            else:
                all_files = [os.path.join(resolved_folder_path, f) for f in os.listdir(resolved_folder_path)]
                image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(tuple(img_extensions_lower))]

            if not image_paths:
                # No images found, return a hash reflecting this state and parameters
                return f"no_images_in_{resolved_folder_path}_{include_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"

            # --- Calculate Hash based on file paths and mtimes ---
            image_paths = sorted(image_paths) # Sort for consistent hashing regardless of OS list order
            mtimes = [os.path.getmtime(f) for f in image_paths]
            file_info_tuple = tuple(zip(image_paths, mtimes))

            # Use hashlib for a more robust hash
            hasher = hashlib.sha256()
            hasher.update(str(file_info_tuple).encode('utf-8'))
            file_list_hash = hasher.hexdigest()

            # Include all parameters that affect the *selection* and *processing* of images in the final change indicator
            # The seed only matters if sort is False.
            seed_component = seed if not sort else "sorted"
            return f"{file_list_hash}_{include_subfolders}_{n_images}_{sort}_{export_with_alpha}_{seed_component}"

        except Exception as e:
             print(f"sokes_nodes.py IS_CHANGED Error in {resolved_folder_path}: {e}")
             # Return a changing value on error to force re-execution
             return float("NaN")

# END Load Random Image with Path and Mask | Sokes 🦬 (with subfolder search)
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

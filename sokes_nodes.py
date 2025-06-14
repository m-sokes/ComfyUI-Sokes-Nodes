import sys
import os
from datetime import datetime
import re
import random
import hashlib
import imghdr
import glob
import torch
import numpy as np
import cv2 # Not used directly in all snippets, but kept from original for broader context
from PIL import Image, ImageOps

import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

from .sokes_color_maps import css3_names_to_hex, css3_hex_to_names, human_readable_map, explicit_targets_for_comparison

if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: a.item()

# --- ComfyUI Integration Imports & Preview Logic ---
try:
    import folder_paths
    # print("sokes_nodes.py: Successfully imported 'folder_paths'.")
except ImportError:
    # print("sokes_nodes.py: 'folder_paths' module not found. Path resolution might be limited.")
    folder_paths = None

preview_available = False
PromptServer = None
nodes_module = None # Renamed to avoid conflict with class name 'nodes' if any
comfy_utils_available = False

# print("sokes_nodes.py: Attempting to import ComfyUI specific modules for previews...")
try:
    from server import PromptServer
    # print("sokes_nodes.py:  - Successfully imported 'PromptServer' from 'server'.")
except ImportError as e:
    # print(f"sokes_nodes.py:  - FAILED to import 'PromptServer' from 'server'. Error: {e}")
    pass # Fail silently for now, checks below will handle
except Exception as e:
    # print(f"sokes_nodes.py:  - FAILED to import 'PromptServer' from 'server' with OTHER error: {e}")
    pass

try:
    import comfy.utils
    comfy_utils_available = True
    # print("sokes_nodes.py:  - Successfully imported 'comfy.utils' module.")
    # try:
    #     from comfy.utils import FONT_PATH # This specific import might fail, but it's not critical
    #     #print("sokes_nodes.py:    - Successfully imported 'FONT_PATH' from 'comfy.utils'.")
    # except ImportError:
    #     #print("sokes_nodes.py:    - Note: 'FONT_PATH' not found in 'comfy.utils' (this is okay for previews).")
    #     pass
except ImportError as e:
    # print(f"sokes_nodes.py:  - FAILED to import 'comfy.utils' module. Error: {e}")
    pass
except Exception as e:
    # print(f"sokes_nodes.py:  - FAILED to import 'comfy.utils' module with OTHER error: {e}")
    pass

try:
    import nodes # This is ComfyUI's 'nodes.py'
    nodes_module = nodes
    # print("sokes_nodes.py:  - Successfully imported ComfyUI 'nodes' module.")
except ImportError as e:
    # print(f"sokes_nodes.py:  - FAILED to import ComfyUI 'nodes' module. Error: {e}")
    pass
except Exception as e:
    # print(f"sokes_nodes.py:  - FAILED to import ComfyUI 'nodes' module with OTHER error: {e}")
    pass

if PromptServer and comfy_utils_available and nodes_module:
    try:
        PromptServer.instance # Crucial check: is the server actually running and instance available?
        preview_available = True
        # print("sokes_nodes.py: Preview system checks passed. Previews should be available.")
    except AttributeError:
        # print("sokes_nodes.py: 'PromptServer.instance' not found (AttributeError). Previews will be disabled.")
        preview_available = False
    except Exception as e:
        # print(f"sokes_nodes.py: 'PromptServer.instance' check failed ({type(e).__name__}: {e}). Previews will be disabled.")
        preview_available = False
# else: # Simplified console output for brevity during normal operation
    # missing_components_msg = []
    # if not PromptServer: missing_components_msg.append("'PromptServer'")
    # if not comfy_utils_available: missing_components_msg.append("'comfy.utils' module")
    # if not nodes_module: missing_components_msg.append("ComfyUI 'nodes' module")
    # if missing_components_msg:
        # print(f"sokes_nodes.py: One or more ComfyUI components for previews failed to import ({', '.join(missing_components_msg)}). Previews disabled.")
    # If components imported but PromptServer.instance failed, that message was already printed (if uncommented).

# --- End ComfyUI Integration Imports & Preview Logic ---


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
        formatted = re.sub(r'(?i)(y+|m+|d+)', lambda m: m.group().upper(), date_format)
        formatted = formatted.replace("YYYY", now.strftime("%Y"))
        formatted = formatted.replace("YY", now.strftime("%y"))
        formatted = formatted.replace("MM", now.strftime("%m"))
        formatted = formatted.replace("M", now.strftime("%m").lstrip("0"))
        formatted = formatted.replace("DD", now.strftime("%d"))
        formatted = formatted.replace("D", now.strftime("%d").lstrip("0"))
        return (formatted,)

    @classmethod
    def IS_CHANGED(cls, date_format):
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
                "latent_1": ("LATENT",), "latent_2": ("LATENT",), "latent_3": ("LATENT",),
                "latent_4": ("LATENT",), "latent_5": ("LATENT",), "latent_6": ("LATENT",),
                "latent_7": ("LATENT",), "latent_8": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "latent_input_switch_9x_sokes"
    OUTPUT_NODE = True
    CATEGORY = "Sokes 🦬"

    def latent_input_switch_9x_sokes(self, latent_select, latent_0, latent_1=None, latent_2=None, latent_3=None, latent_4=None, latent_5=None, latent_6=None, latent_7=None, latent_8=None):
        latent_map = {
            0: latent_0, 1: latent_1, 2: latent_2, 3: latent_3, 4: latent_4,
            5: latent_5, 6: latent_6, 7: latent_7, 8: latent_8
        }
        select_idx = int(round(latent_select))
        selected_latent = latent_map.get(select_idx) # Get will return None if key doesn't exist

        if selected_latent is not None:
            return (selected_latent,)
        else:
            # Fallback to latent_0 if selection is invalid or selected latent is None
            # (and latent_0 itself is not None, though it's required)
            if select_idx not in latent_map:
                 print(f"latent_input_switch_9x_sokes: Invalid latent_select index {select_idx}. Defaulting to latent_0.")
            elif select_idx != 0 : # only print if not trying to select a None optional input (and it wasn't latent_0)
                 print(f"latent_input_switch_9x_sokes: latent_{select_idx} is None. Defaulting to latent_0.")
            return (latent_0,)

# END Latent Input Swtich x9 | Sokes 🦬
##############################################################

##############################################################
# START Replace Text with Regex | Sokes 🦬

class replace_text_regex_sokes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "regex_pattern": ("STRING", {"multiline": False}),
            "new": ("STRING", {"multiline": False})
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fn_replace_text_regex_sokes"
    CATEGORY = "Sokes 🦬"

    def fn_replace_text_regex_sokes(self, regex_pattern, new, text):
        return (re.sub(regex_pattern, new, text),)

# END Replace Text with Regex | Sokes 🦬
##############################################################

##############################################################
# START Load Random Image with Path and Mask | Sokes 🦬

class load_random_image_sokes:
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory() if folder_paths else "temp_sokes_previews"
        self.type = "temp"
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                print(f"sokes_nodes.py load_random_image_sokes: Warning: Could not create temp directory {self.output_dir}: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": True}),
                "filename_optional": ("STRING", {"default": "", "multiline": False}),
                "search_subfolders": ("BOOLEAN", {"default": False}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sort": ("BOOLEAN", {"default": False}),
                "export_with_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image_or_file"
    OUTPUT_NODE = True

    def _get_all_matching_images(self, folder_path, filename_optional, search_subfolders):
        """Helper to find all image files based on paths, wildcards, and pipe delimiters."""
        image_paths_found = set()
        folder_paths_list = [p.strip() for p in folder_path.split('|')]

        for p in folder_paths_list:
            if not p and not filename_optional:
                continue

            pattern = os.path.join(p, filename_optional)
            try:
                # Enable recursive globbing with '**', relies on CWD being ComfyUI root
                glob_matches = glob.glob(pattern, recursive=True)
            except Exception as e:
                print(f"sokes_nodes.py: Warning - glob pattern '{pattern}' is invalid: {e}")
                continue

            for match in glob_matches:
                abs_match = os.path.abspath(match)
                if not os.path.exists(abs_match):
                    continue

                if os.path.isfile(abs_match):
                    if any(abs_match.lower().endswith(ext) for ext in self.IMG_EXTENSIONS):
                        image_paths_found.add(os.path.normpath(abs_match))
                elif os.path.isdir(abs_match):
                    try:
                        images_in_dir = self.find_image_files(abs_match, search_subfolders)
                        image_paths_found.update(images_in_dir)
                    except Exception as e:
                        print(f"sokes_nodes.py: Warning - Could not search directory '{abs_match}': {e}")
        return list(image_paths_found)

    def find_image_files(self, folder_path_abs: str, search_subfolders: bool):
        image_paths = []
        img_extensions_lower = [ext.lower() for ext in self.IMG_EXTENSIONS]
        
        if not os.path.isdir(folder_path_abs):
            raise FileNotFoundError(f"Directory not found for listing: {folder_path_abs}")

        try:
            if search_subfolders:
                for root, _, files in os.walk(folder_path_abs):
                    for file_name_in_walk in files:
                        if file_name_in_walk.lower().endswith(tuple(img_extensions_lower)):
                           full_path = os.path.join(root, file_name_in_walk) # root from os.walk is already absolute
                           if os.path.isfile(full_path):
                               image_paths.append(os.path.normpath(full_path))
            else:
                for f_name in os.listdir(folder_path_abs):
                    full_path = os.path.join(folder_path_abs, f_name)
                    if os.path.isfile(full_path) and f_name.lower().endswith(tuple(img_extensions_lower)):
                        image_paths.append(os.path.normpath(full_path))
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder_path_abs}': {e}")
        return image_paths

    def validate_and_load_image(self, image_path_abs: str, final_image_mode: str):
        try:
            img_pil = Image.open(image_path_abs)
            img_pil = ImageOps.exif_transpose(img_pil)

            actual_format = imghdr.what(image_path_abs)
            if not actual_format:
                try: img_pil.verify()
                except Exception as pil_e:
                    raise ValueError(f"Invalid/corrupt image (PIL verify): {os.path.basename(image_path_abs)} - {pil_e}")
            
            converted_image_pil = img_pil.convert(final_image_mode) if img_pil.mode != final_image_mode else img_pil
            image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            mask_tensor = None
            if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
                # If exporting RGB but original has alpha, get alpha from original's RGBA conversion
                # If exporting RGBA, get alpha from the final converted image
                alpha_source_pil = img_pil.convert('RGBA') if final_image_mode == 'RGB' else converted_image_pil
                mask_pil_channel = alpha_source_pil.split()[-1]
                mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]
            else:
                mask_shape = (image_tensor.shape[1], image_tensor.shape[2])
                mask_np = np.ones(mask_shape, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_np)[None,]
            return image_tensor, mask_tensor, converted_image_pil
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {os.path.basename(image_path_abs)}")
        except ValueError as ve: raise ve # Re-raise specific validation errors
        except Exception as e:
            raise RuntimeError(f"Error loading/processing image {os.path.basename(image_path_abs)}: {e}")

    def load_image_or_file(self, folder_path, filename_optional, search_subfolders, n_images, seed, sort, export_with_alpha):
        # Use the helper to get all potential image paths from wildcards and multiple folders
        image_paths_found_normalized = self._get_all_matching_images(folder_path, filename_optional, search_subfolders)

        # Validate and filter corrupt images from the list
        valid_image_paths = []
        for f_path_norm in image_paths_found_normalized:
            try:
                is_hdr = imghdr.what(f_path_norm)
                if is_hdr:
                    valid_image_paths.append(f_path_norm)
                    continue
                else:
                    with Image.open(f_path_norm) as img_test: img_test.verify()
                    valid_image_paths.append(f_path_norm)
            except Exception:
                if os.path.exists(f_path_norm):
                    print(f"sokes_nodes.py: Skipping invalid/corrupt file: {os.path.basename(f_path_norm)}")
        
        if not valid_image_paths:
            raise FileNotFoundError(f"No valid images found for folder='{folder_path}' and filename='{filename_optional}'. Check paths, wildcards, and permissions.")

        # Selection logic (random/sorted)
        num_available = len(valid_image_paths)
        actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
        if actual_n_images == 0: raise ValueError("Zero images to load after filtering valid images.")
        
        if actual_n_images < n_images and n_images > 0:
              print(f"sokes_nodes.py: Warning: Requested {n_images} images, but only {num_available} were valid. Loading {actual_n_images}.")

        selected_paths_abs = []
        if not sort:
            random.seed(seed)
            random.shuffle(valid_image_paths)
            selected_paths_abs = valid_image_paths[:actual_n_images]
        else:
            def natural_sort_key(s_path):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', os.path.basename(s_path))]
            valid_image_paths_sorted = sorted(valid_image_paths, key=natural_sort_key)
            
            start_python_index = 0
            if seed > 0 and num_available > 0:
                start_python_index = (seed - 1) % num_available
            
            selected_paths_abs = [valid_image_paths_sorted[(start_python_index + i) % num_available] for i in range(actual_n_images)]

        if not selected_paths_abs: raise ValueError("No images were selected to load for processing.")

        # Batch loading logic
        output_images_tensor_list, output_masks_tensor_list = [], []
        loaded_paths_final_abs, pil_images_for_preview = [], []
        
        final_image_mode = "RGB"
        if export_with_alpha:
            for image_path_check_abs in selected_paths_abs:
                try:
                    with Image.open(image_path_check_abs) as img_pil_check:
                         if img_pil_check.mode in ('RGBA', 'LA') or \
                            (img_pil_check.mode == 'P' and 'transparency' in img_pil_check.info):
                            final_image_mode = "RGBA"; break
                except: pass # Ignore errors during this pre-check
        
        if export_with_alpha and final_image_mode == "RGB" and selected_paths_abs: # Only print if relevant
            print(f"sokes_nodes.py: Note: export_with_alpha=True, but no images with alpha found in selection. Outputting RGB.")

        first_image_shape_hwc = None
        for image_path_abs_current in selected_paths_abs:
            try:
                image_tensor, mask_tensor, loaded_pil_image = self.validate_and_load_image(image_path_abs_current, final_image_mode)
                
                current_shape_hwc = image_tensor.shape[1:4] # H, W, C
                if first_image_shape_hwc is None:
                    first_image_shape_hwc = current_shape_hwc
                elif current_shape_hwc != first_image_shape_hwc and len(selected_paths_abs) > 1:
                    print(f"sokes_nodes.py: ⚠️ Warning: Image {os.path.basename(image_path_abs_current)} dims/chans ({current_shape_hwc}) "
                          f"differ from first image ({first_image_shape_hwc}). Batch may be inconsistent.")

                output_images_tensor_list.append(image_tensor)
                output_masks_tensor_list.append(mask_tensor)
                loaded_paths_final_abs.append(image_path_abs_current)
                pil_images_for_preview.append(loaded_pil_image)
            except Exception as e:
                print(f"sokes_nodes.py: ❌ Skipping image {os.path.basename(image_path_abs_current)} due to error: {str(e)}")
                continue

        if not output_images_tensor_list: raise ValueError("No images were successfully loaded into tensors.")
        
        final_image_batch = torch.cat(output_images_tensor_list, dim=0)
        final_mask_batch = torch.cat(output_masks_tensor_list, dim=0)
        
        previews_out_list = []
        if preview_available and pil_images_for_preview:
            preview_subfolder_name = "sokes_nodes_previews"
            full_preview_output_folder = os.path.join(self.output_dir, preview_subfolder_name)
            if not os.path.exists(full_preview_output_folder):
                try: os.makedirs(full_preview_output_folder, exist_ok=True)
                except: print(f"sokes_nodes.py: Error creating preview subfolder {full_preview_output_folder}. Previews may fail.")

            for i, pil_img in enumerate(pil_images_for_preview):
                try:
                    unique_hash = hashlib.sha1(loaded_paths_final_abs[i].encode('utf-8')).hexdigest()[:10]
                    preview_filename = f"preview_{unique_hash}_{i}.png"
                    filepath = os.path.join(full_preview_output_folder, preview_filename)
                    pil_img.save(filepath, compress_level=4)
                    previews_out_list.append({
                        "filename": preview_filename,
                        "subfolder": preview_subfolder_name,
                        "type": self.type
                    })
                except Exception as e_prev:
                    print(f"sokes_nodes.py: Error generating preview for {os.path.basename(loaded_paths_final_abs[i])}: {e_prev}")
        
        return {"ui": {"images": previews_out_list}, "result": (final_image_batch, final_mask_batch, loaded_paths_final_abs)}

    @classmethod
    def IS_CHANGED(cls, folder_path, filename_optional, search_subfolders, n_images, seed, sort, export_with_alpha):
        instance = cls()
        try:
            image_paths = instance._get_all_matching_images(folder_path, filename_optional, search_subfolders)
        except Exception as e:
            return f"path_error_{folder_path}_{filename_optional}_{e}"

        if not image_paths:
            return f"no_files_{folder_path}_{filename_optional}_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"

        mtimes = []
        for p in image_paths:
            try:
                mtimes.append(os.path.getmtime(p))
            except OSError:
                pass
        
        hasher = hashlib.sha256()
        hasher.update(str(sorted(image_paths)).encode('utf-8'))
        hasher.update(str(sorted(mtimes)).encode('utf-8'))
        
        params_string = f"_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}_{filename_optional}"
        hasher.update(params_string.encode('utf-8'))
        
        return hasher.hexdigest()

# END Load Random Image with Path and Mask | Sokes 🦬
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
                "hex_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            },
            "optional": {
                "use_css_name": ("BOOLEAN", {"default": False})
            }
        }

    def hex_to_color_name_fn(self, hex_color, use_css_name=False):
        if not hex_color: return ("Input hex color is empty.",)
        hex_color_proc = hex_color.strip()
        if not hex_color_proc.startswith("#"): hex_color_proc = "#" + hex_color_proc
        if len(hex_color_proc) == 4:
            hex_color_proc = f"#{hex_color_proc[1]*2}{hex_color_proc[2]*2}{hex_color_proc[3]*2}"
        if len(hex_color_proc) != 7: return (f"Invalid hex format: {hex_color} (processed: {hex_color_proc})",)

        try:
            standard_name = webcolors.hex_to_name(hex_color_proc, spec="css3")
            final_color_name = standard_name
            if not use_css_name:
                 final_color_name = human_readable_map.get(standard_name, standard_name)
            return (final_color_name,)
        except ValueError:
            try:
                requested_rgb_clamped = tuple(max(0, min(255, c)) for c in webcolors.hex_to_rgb(hex_color_proc))
                requested_lab = convert_color(sRGBColor(*requested_rgb_clamped, is_upscaled=True), LabColor)
            except Exception as e:
                return (f"Invalid input hex '{hex_color_proc}' or conversion error: {e}",)

            min_dist = float('inf')
            closest_name_internal = None
            
            color_sources = [
                explicit_targets_for_comparison, # Check explicit targets first
                css3_names_to_hex              # Then CSS3 names
            ]

            for source_map in color_sources:
                for name_key, hex_val_orig in source_map.items():
                    current_hex_proc = hex_val_orig.strip()
                    if not current_hex_proc.startswith("#"): current_hex_proc = "#" + current_hex_proc
                    if len(current_hex_proc) == 4:
                        current_hex_proc = f"#{current_hex_proc[1]*2}{current_hex_proc[2]*2}{current_hex_proc[3]*2}"
                    if len(current_hex_proc) != 7: continue

                    try:
                        cand_rgb_clamped = tuple(max(0, min(255, c)) for c in webcolors.hex_to_rgb(current_hex_proc))
                        cand_lab = convert_color(sRGBColor(*cand_rgb_clamped, is_upscaled=True), LabColor)
                        d = delta_e_cie2000(requested_lab, cand_lab)
                        if d < min_dist:
                            min_dist = d
                            closest_name_internal = name_key 
                    except Exception: # Ignore errors for individual candidates
                        pass 
            
            if closest_name_internal is None: return ("Could not find any closest color match.",)

            if use_css_name:
                final_color_name = closest_name_internal # Could be from explicit_targets or css3_names_to_hex
                # If it was from explicit_targets, and a true CSS name is desired, this might need adjustment
                # or we assume explicit_targets keys are "CSS-like enough" if use_css_name is true.
            else:
                final_color_name = human_readable_map.get(closest_name_internal, closest_name_internal)
            
            return (final_color_name,)

# END: Hex to Color Name | Sokes 🦬
##############################################################


##############################################################
# START Random Number | Sokes 🦬
class random_number_sokes:
    CATEGORY = "Sokes 🦬"
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
        min_val = min(minimum, maximum)
        max_val = max(minimum, maximum)
        random.seed(seed)
        primary_float = random.uniform(min_val, max_val)
        derived_int = int(round(primary_float))
        midpoint = min_val + (max_val - min_val) / 2.0
        derived_bool = primary_float > midpoint
        return (derived_int, primary_float, derived_bool)

    @classmethod
    def IS_CHANGED(cls, minimum, maximum, seed):
        h = hashlib.sha256()
        # Ensure consistent string representation for hashing
        h.update(f"min:{minimum}-max:{maximum}-seed:{seed}".encode('utf-8'))
        return h.hexdigest()
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
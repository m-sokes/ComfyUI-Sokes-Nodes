import sys
import os
from datetime import datetime

# Environment Variables Required:
# GOOGLE_STREET_VIEW_API_KEY - Your Google Street View API key
# RUNPOD_API_KEY - Your Runpod API key (optional, for Runpod integration)
import re
import random
import hashlib
import imghdr
import glob
import json
import time
import base64
import torch
import numpy as np
import cv2 # Not used directly in all snippets, but kept from original for broader context
from PIL import Image, ImageOps, ImageDraw
import math
import requests
import urllib.parse
import io
from typing import Dict, Any, Tuple

import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

from .sokes_color_maps import css3_names_to_hex, css3_hex_to_names, human_readable_map, explicit_targets_for_comparison

# --- Google Street View API Key ---
# Get your Google Street View API key from environment variable GOOGLE_STREET_VIEW_API_KEY
# You can get one from the Google Cloud Platform: https://console.cloud.google.com/marketplace/product/google/street-view-image-backend.googleapis.com
GOOGLE_STREET_VIEW_API_KEY = os.getenv('GOOGLE_STREET_VIEW_API_KEY')

if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: a.item()

# --- ComfyUI Integration Imports & Preview Logic ---
try:
    import folder_paths
except ImportError:
    folder_paths = None

preview_available = False
PromptServer = None
nodes_module = None 
comfy_utils_available = False
api_routes_setup = False # Global flag to ensure routes are set up only once

try:
    from server import PromptServer
    from aiohttp import web
except ImportError as e:
    pass 

try:
    import comfy.utils
    comfy_utils_available = True
except ImportError as e:
    pass

try:
    import nodes 
    nodes_module = nodes
except ImportError as e:
    pass

if PromptServer and comfy_utils_available and nodes_module:
    try:
        PromptServer.instance 
        preview_available = True
    except AttributeError:
        preview_available = False
    except Exception as e:
        preview_available = False
# --- End ComfyUI Integration Imports & Preview Logic ---


# --- Helper function for finding images, used by multiple nodes ---
def find_image_files_in_folder(folder_path_abs: str, search_subfolders: bool):
    image_paths = []
    img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]
    img_extensions_lower = [ext.lower() for ext in img_extensions]
    if not os.path.isdir(folder_path_abs):
        return []
    try:
        if search_subfolders:
            for root, _, files in os.walk(folder_path_abs):
                for file_name_in_walk in files:
                    if file_name_in_walk.lower().endswith(tuple(img_extensions_lower)):
                        full_path = os.path.join(root, file_name_in_walk)
                        if os.path.isfile(full_path): image_paths.append(os.path.normpath(full_path))
        else:
            for f_name in os.listdir(folder_path_abs):
                full_path = os.path.join(folder_path_abs, f_name)
                if os.path.isfile(full_path) and f_name.lower().endswith(tuple(img_extensions_lower)):
                    image_paths.append(os.path.normpath(full_path))
    except Exception as e:
        print(f"sokes_nodes.py: Error listing files in folder '{folder_path_abs}': {e}")
    return image_paths

# --- API Endpoint for Image Picker ---
if preview_available and not api_routes_setup:
    @PromptServer.instance.routes.get("/sokes/get_image_list")
    async def get_image_list(request):
        request_path = request.query.get("folder_path", "")
        folder_path = request_path

        if not folder_path:
            return web.json_response({"error": "No path provided."}, status=400)

        if os.path.isfile(folder_path):
            folder_path = os.path.dirname(folder_path)

        if not os.path.isdir(folder_path):
            return web.json_response({"error": "Folder not found or path is invalid.", "path_received": request_path}, status=404)

        try:
            image_files = find_image_files_in_folder(folder_path, False)
            
            def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
            image_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))

            input_dir = folder_paths.get_input_directory()
            output_dir = folder_paths.get_output_directory()
            temp_dir = folder_paths.get_temp_directory()

            formatted_files = []
            for full_path in image_files:
                full_path_norm = os.path.abspath(full_path)
                subfolder, type = "", ""

                if full_path_norm.startswith(os.path.abspath(input_dir)):
                    type = "input"
                    subfolder_path = os.path.relpath(os.path.dirname(full_path_norm), input_dir)
                    subfolder = '' if subfolder_path == '.' else subfolder_path
                elif full_path_norm.startswith(os.path.abspath(output_dir)):
                    type = "output"
                    subfolder_path = os.path.relpath(os.path.dirname(full_path_norm), output_dir)
                    subfolder = '' if subfolder_path == '.' else subfolder_path
                elif full_path_norm.startswith(os.path.abspath(temp_dir)):
                    type = "temp"
                    subfolder_path = os.path.relpath(os.path.dirname(full_path_norm), temp_dir)
                    subfolder = '' if subfolder_path == '.' else subfolder_path

                if type:
                    filename = os.path.basename(full_path_norm)
                    formatted_files.append({"filename": filename, "subfolder": subfolder, "type": type, "full_path": full_path_norm})
            
            return web.json_response(formatted_files)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
            
    api_routes_setup = True


##############################################################
# START Image Picker | Sokes 🦬

class ImagePickerSokes:
    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": folder_paths.get_input_directory() if folder_paths else ""}),
                "current_image": ("STRING", {"default": "", "widget": "hidden"}),
                # CORRECTED: Default is now False
                "always_pick_last": ("BOOLEAN", {"default": False}),
            }
        }

    def validate_and_load_image(self, image_path_abs: str):
        try:
            img_pil = Image.open(image_path_abs); img_pil = ImageOps.exif_transpose(img_pil)
            converted_image_pil = img_pil.convert("RGB")
            image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            if 'A' in img_pil.getbands():
                mask_tensor = torch.from_numpy(np.array(img_pil.getchannel('A')).astype(np.float32) / 255.0)[None,]
            else:
                mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)

            return image_tensor, mask_tensor
        except Exception as e:
            raise RuntimeError(f"Error loading or processing image {os.path.basename(image_path_abs)}: {e}")

    def execute(self, folder_path, current_image, always_pick_last):
        image_to_load = ""
        final_folder_path = folder_path

        if os.path.isfile(final_folder_path):
            final_folder_path = os.path.dirname(final_folder_path)

        if always_pick_last:
            if not final_folder_path or not os.path.isdir(final_folder_path):
                raise FileNotFoundError(f"Folder path for 'always_pick_last' is invalid or not found: {final_folder_path}")
            
            image_files = find_image_files_in_folder(final_folder_path, False)
            if not image_files:
                raise FileNotFoundError(f"No images found in the specified folder: {final_folder_path}")

            def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
            image_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
            image_to_load = image_files[-1]
        else:
            image_to_load = current_image

        if not image_to_load or not os.path.isfile(image_to_load):
            print(f"sokes_nodes.py Image Picker: No valid image selected or file not found ('{image_to_load}'). Returning a default black image.")
            black_image_tensor = torch.zeros(1, 64, 64, 3)
            black_mask_tensor = torch.ones(1, 64, 64)
            return (black_image_tensor, black_mask_tensor, "")

        try:
            image_tensor, mask_tensor = self.validate_and_load_image(image_to_load)
            return (image_tensor, mask_tensor, image_to_load)
        except Exception as e:
            print(f"sokes_nodes.py Image Picker: Error loading image '{image_to_load}': {e}. Returning a default black image.")
            black_image_tensor = torch.zeros(1, 64, 64, 3)
            black_mask_tensor = torch.ones(1, 64, 64)
            return (black_image_tensor, black_mask_tensor, "")

    @classmethod
    def IS_CHANGED(cls, folder_path, current_image, always_pick_last):
        mtime = os.path.getmtime(current_image) if current_image and os.path.exists(current_image) else ""
        return f"{folder_path}:{current_image}:{always_pick_last}:{mtime}"

# END Image Picker | Sokes 🦬
##############################################################


##############################################################
# START Current Date & Time | Sokes 🦬

class current_date_time_sokes:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "date_time_format": ("STRING", {"default": 'YYYY-MM-DD', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_date_time",)
    FUNCTION = "current_date_time_sokes"
    CATEGORY = "Sokes 🦬"

    def current_date_time_sokes(self, date_time_format):
        now = datetime.now()
        temp_format = date_time_format
        temp_format = re.sub('YYYY', '%Y', temp_format, flags=re.IGNORECASE)
        temp_format = re.sub('YY',  '%y', temp_format, flags=re.IGNORECASE)
        temp_format = re.sub('MM',  '%m', temp_format, flags=re.IGNORECASE)
        temp_format = re.sub('DD',  '%d', temp_format, flags=re.IGNORECASE)
        temp_format = re.sub(r'(?<!%)\bM\b', str(now.month), temp_format, flags=re.IGNORECASE)
        temp_format = re.sub(r'(?<!%)\bD\b', str(now.day), temp_format, flags=re.IGNORECASE)
        final_string = now.strftime(temp_format)
        return (final_string,)

    @classmethod
    def IS_CHANGED(cls, date_time_format):
        return (datetime.now().timestamp(),)

# END Current Date & Time | Sokes 🦬
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
        latent_map = { 0: latent_0, 1: latent_1, 2: latent_2, 3: latent_3, 4: latent_4, 5: latent_5, 6: latent_6, 7: latent_7, 8: latent_8 }
        select_idx = int(round(latent_select))
        selected_latent = latent_map.get(select_idx)
        if selected_latent is not None:
            return (selected_latent,)
        else:
            if select_idx not in latent_map: print(f"latent_input_switch_9x_sokes: Invalid latent_select index {select_idx}. Defaulting to latent_0.")
            elif select_idx != 0 : print(f"latent_input_switch_9x_sokes: latent_{select_idx} is None. Defaulting to latent_0.")
            return (latent_0,)

# END Latent Input Swtich x9 | Sokes 🦬
##############################################################

##############################################################
# START Replace Text with Regex | Sokes 🦬

class replace_text_regex_sokes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}), "regex_pattern": ("STRING", {"multiline": False}), "new": ("STRING", {"multiline": False}) }}
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
            try: os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e: print(f"sokes_nodes.py load_random_image_sokes: Warning: Could not create temp directory {self.output_dir}: {e}")
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "folder_path": ("STRING", {"default": "", "multiline": True}), "filename_optional": ("STRING", {"default": "", "multiline": False}), "search_subfolders": ("BOOLEAN", {"default": False}), "n_images": ("INT", {"default": 1, "min": -1, "max": 100}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), "sort": ("BOOLEAN", {"default": False}), "export_with_alpha": ("BOOLEAN", {"default": False}), } }
    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST"); RETURN_NAMES = ("image", "mask", "image_path"); FUNCTION = "load_image_or_file"; OUTPUT_NODE = True
    def _get_all_matching_images(self, folder_path, filename_optional, search_subfolders):
        image_paths_found = []
        path_string = folder_path.replace('\n', '|'); folder_paths_list = [p.strip() for p in path_string.split('|') if p.strip()]
        for p in folder_paths_list:
            if os.path.isfile(p):
                if any(p.lower().endswith(ext) for ext in self.IMG_EXTENSIONS): image_paths_found.append(os.path.normpath(os.path.abspath(p)))
                continue
            pattern = os.path.join(p, filename_optional)
            try: glob_matches = glob.glob(pattern, recursive=True)
            except Exception as e: print(f"sokes_nodes.py: Warning - glob pattern '{pattern}' is invalid: {e}"); continue
            for match in glob_matches:
                abs_match = os.path.abspath(match)
                if not os.path.exists(abs_match): continue
                if os.path.isfile(abs_match):
                    if any(abs_match.lower().endswith(ext) for ext in self.IMG_EXTENSIONS): image_paths_found.append(os.path.normpath(abs_match))
                elif os.path.isdir(abs_match):
                    try: images_in_dir = find_image_files_in_folder(abs_match, search_subfolders); image_paths_found.extend(images_in_dir)
                    except Exception as e: print(f"sokes_nodes.py: Warning - Could not search directory '{abs_match}': {e}")
        return image_paths_found
    
    def validate_and_load_image(self, image_path_abs: str, final_image_mode: str):
        try:
            img_pil = Image.open(image_path_abs); img_pil = ImageOps.exif_transpose(img_pil)
            actual_format = imghdr.what(image_path_abs)
            if not actual_format:
                try: img_pil.verify()
                except Exception as pil_e: raise ValueError(f"Invalid/corrupt image (PIL verify): {os.path.basename(image_path_abs)} - {pil_e}")
            converted_image_pil = img_pil.convert(final_image_mode) if img_pil.mode != final_image_mode else img_pil
            image_np = np.array(converted_image_pil).astype(np.float32) / 255.0; image_tensor = torch.from_numpy(image_np)[None,]
            if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
                alpha_source_pil = img_pil.convert('RGBA') if final_image_mode == 'RGB' else converted_image_pil
                mask_pil_channel = alpha_source_pil.split()[-1]
                mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0; mask_tensor = torch.from_numpy(mask_np)[None,]
            else:
                mask_shape = (image_tensor.shape[1], image_tensor.shape[2])
                mask_np = np.ones(mask_shape, dtype=np.float32); mask_tensor = torch.from_numpy(mask_np)[None,]
            return image_tensor, mask_tensor, converted_image_pil
        except FileNotFoundError: raise FileNotFoundError(f"Image not found: {os.path.basename(image_path_abs)}")
        except ValueError as ve: raise ve
        except Exception as e: raise RuntimeError(f"Error loading/processing image {os.path.basename(image_path_abs)}: {e}")
    
    def load_image_or_file(self, folder_path, filename_optional, search_subfolders, n_images, seed, sort, export_with_alpha):
        if not folder_path or not folder_path.strip():
            print("sokes_nodes.py Load Random Image: folder_path is empty. Returning a default black image.")
            black_image_pil = Image.new('RGB', (64, 64), (0, 0, 0))
            image_np = np.array(black_image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            mask_np = np.ones((64, 64), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_np)[None,]
            return (image_tensor, mask_tensor, [""])

        image_paths_found_normalized = self._get_all_matching_images(folder_path, filename_optional, search_subfolders)
        unique_paths_for_validation = sorted(list(set(image_paths_found_normalized)))
        valid_image_paths_set = set()
        for f_path_norm in unique_paths_for_validation:
            try:
                with Image.open(f_path_norm) as img_test: img_test.verify()
                valid_image_paths_set.add(f_path_norm)
            except Exception:
                if os.path.exists(f_path_norm): print(f"sokes_nodes.py: Skipping invalid/corrupt file: {os.path.basename(f_path_norm)}")
        final_selection_pool = [p for p in image_paths_found_normalized if p in valid_image_paths_set]
        if not final_selection_pool: raise FileNotFoundError(f"No valid images found for folder='{folder_path}' and filename='{filename_optional}'. Check paths, wildcards, and permissions.")
        
        num_available = len(final_selection_pool)
        selected_paths_abs = []
        def natural_sort_key(s_path): return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', os.path.basename(s_path))]

        is_get_last_mode = (n_images == -1)
        should_sort = sort or is_get_last_mode

        if should_sort:
            final_selection_pool.sort(key=natural_sort_key)

        if is_get_last_mode:
            if num_available > 0:
                selected_paths_abs = [final_selection_pool[-1]]
        else:
            actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
            if actual_n_images == 0: raise ValueError("Zero images to load after filtering valid images.")
            if actual_n_images < n_images and n_images > 0: print(f"sokes_nodes.py: Warning: Requested {n_images} images, but only {num_available} were in the weighted pool. Loading {actual_n_images}.")
    
            if should_sort:
                start_python_index = 0
                if seed > 0 and num_available > 0: start_python_index = (seed - 1) % num_available
                selected_paths_abs = [final_selection_pool[(start_python_index + i) % num_available] for i in range(actual_n_images)]
            else:
                random.seed(seed)
                random.shuffle(final_selection_pool)
                selected_paths_abs = final_selection_pool[:actual_n_images]

        if not selected_paths_abs: raise ValueError("No images were selected to load for processing.")
        output_images_tensor_list, output_masks_tensor_list = [], []; loaded_paths_final_abs, pil_images_for_preview = [], []
        final_image_mode = "RGB"
        if export_with_alpha:
            for image_path_check_abs in list(set(selected_paths_abs)):
                try:
                    with Image.open(image_path_check_abs) as img_pil_check:
                         if img_pil_check.mode in ('RGBA', 'LA') or (img_pil_check.mode == 'P' and 'transparency' in img_pil_check.info): final_image_mode = "RGBA"; break
                except: pass
        if export_with_alpha and final_image_mode == "RGB" and selected_paths_abs: print(f"sokes_nodes.py: Note: export_with_alpha=True, but no images with alpha found in selection. Outputting RGB.")
        first_image_shape_hwc = None; warned_about_shapes = set()
        for image_path_abs_current in selected_paths_abs:
            try:
                image_tensor, mask_tensor, loaded_pil_image = self.validate_and_load_image(image_path_abs_current, final_image_mode)
                current_shape_hwc = image_tensor.shape[1:4]
                if first_image_shape_hwc is None: first_image_shape_hwc = current_shape_hwc
                elif current_shape_hwc != first_image_shape_hwc and image_path_abs_current not in warned_about_shapes:
                    print(f"sokes_nodes.py: ⚠️ Warning: Image {os.path.basename(image_path_abs_current)} dims/chans ({current_shape_hwc}) differ from first image ({first_image_shape_hwc}). Batch may be inconsistent.")
                    warned_about_shapes.add(image_path_abs_current)
                output_images_tensor_list.append(image_tensor); output_masks_tensor_list.append(mask_tensor); loaded_paths_final_abs.append(image_path_abs_current); pil_images_for_preview.append(loaded_pil_image)
            except Exception as e: print(f"sokes_nodes.py: ❌ Skipping image {os.path.basename(image_path_abs_current)} due to error: {str(e)}"); continue
        if not output_images_tensor_list: raise ValueError("No images were successfully loaded into tensors.")
        final_image_batch = torch.cat(output_images_tensor_list, dim=0); final_mask_batch = torch.cat(output_masks_tensor_list, dim=0)
        previews_out_list = []
        if preview_available and pil_images_for_preview:
            preview_subfolder_name = "sokes_nodes_previews"
            full_preview_output_folder = os.path.join(self.output_dir, preview_subfolder_name)
            if not os.path.exists(full_preview_output_folder):
                try: os.makedirs(full_preview_output_folder, exist_ok=True)
                except: print(f"sokes_nodes.py: Error creating preview subfolder {full_preview_output_folder}. Previews may fail.")
            for i, pil_img in enumerate(pil_images_for_preview):
                try:
                    unique_hash = hashlib.sha1(f"{loaded_paths_final_abs[i]}_{i}".encode('utf-8')).hexdigest()[:10]
                    preview_filename = f"preview_{unique_hash}.png"; filepath = os.path.join(full_preview_output_folder, preview_filename)
                    pil_img.save(filepath, compress_level=4)
                    previews_out_list.append({"filename": preview_filename, "subfolder": preview_subfolder_name, "type": self.type})
                except Exception as e_prev: print(f"sokes_nodes.py: Error generating preview for {os.path.basename(loaded_paths_final_abs[i])}: {e_prev}")
        return {"ui": {"images": previews_out_list}, "result": (final_image_batch, final_mask_batch, loaded_paths_final_abs)}
    @classmethod
    def IS_CHANGED(cls, folder_path, filename_optional, search_subfolders, n_images, seed, sort, export_with_alpha):
        if not folder_path or not folder_path.strip(): return f"no_path_{seed}"
        instance = cls()
        try: image_paths = instance._get_all_matching_images(folder_path, filename_optional, search_subfolders)
        except Exception as e: return f"path_error_{folder_path}_{filename_optional}_{e}"
        if not image_paths: return f"no_files_{folder_path}_{filename_optional}_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"
        mtimes = []; unique_paths = sorted(list(set(image_paths)))
        for p in unique_paths:
            try: mtimes.append(os.path.getmtime(p))
            except OSError: pass
        hasher = hashlib.sha256()
        hasher.update(str(sorted(image_paths)).encode('utf-8')); hasher.update(str(sorted(mtimes)).encode('utf-8'))
        params_string = f"_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}_{filename_optional}"
        hasher.update(params_string.encode('utf-8')); return hasher.hexdigest()

# END Load Random Image with Path and Mask | Sokes 🦬
##############################################################


##############################################################
# START ComfyUI Folder Paths | Sokes 🦬

class ComfyUI_folder_paths_sokes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_folder", "input_folder", "models_folder")
    FUNCTION = "get_comfy_paths"
    CATEGORY = "Sokes 🦬/File Paths"
    OUTPUT_NODE = True

    def get_comfy_paths(self):
        if folder_paths is None:
            print("sokes_nodes.py: Warning: ComfyUI folder_paths module not found. Returning empty strings.")
            return ("", "", "")

        output_path = os.path.abspath(folder_paths.get_output_directory())
        input_path = os.path.abspath(folder_paths.get_input_directory())
        models_path = os.path.abspath(folder_paths.models_dir)
        
        return (output_path, input_path, models_path)

    @classmethod
    def IS_CHANGED(cls):
        return "static_sokes_folder_paths_node"

# END ComfyUI Folder Paths | Sokes 🦬
##############################################################


##############################################################
# START Hex to Color Name | Sokes 🦬

class hex_to_color_name_sokes:
    CATEGORY = "Sokes 🦬"
    RETURN_TYPES = ("STRING", "STRING",); RETURN_NAMES = ("color_name", "hex",) ; FUNCTION = "execute"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_color": ("STRING", {"default": "#FF6347, #4682B4, #32CD32", "multiline": True}),
            },
            "optional": {
                "use_css_name": ("BOOLEAN", {"default": False})
            }
        }

    def _find_closest_color_name(self, hex_code, use_css_name):
        try:
            standard_name = webcolors.hex_to_name(hex_code, spec="css3")
            return human_readable_map.get(standard_name, standard_name) if not use_css_name else standard_name
        except ValueError:
            try:
                requested_rgb = webcolors.hex_to_rgb(hex_code)
                requested_lab = convert_color(sRGBColor(*requested_rgb, is_upscaled=True), LabColor)
            except Exception:
                return "Invalid"
            
            min_dist = float('inf')
            closest_name_internal = None
            
            for source_map in [explicit_targets_for_comparison, css3_names_to_hex]:
                for name_key, hex_val_orig in source_map.items():
                    try:
                        cand_rgb = webcolors.hex_to_rgb(hex_val_orig)
                        cand_lab = convert_color(sRGBColor(*cand_rgb, is_upscaled=True), LabColor)
                        dist = delta_e_cie2000(requested_lab, cand_lab)
                        if dist < min_dist:
                            min_dist, closest_name_internal = dist, name_key
                    except Exception:
                        continue
            
            if closest_name_internal:
                return human_readable_map.get(closest_name_internal, closest_name_internal) if not use_css_name else closest_name_internal
            else:
                return "Unknown"

    def execute(self, hex_color, use_css_name=False):
        if not hex_color or not hex_color.strip():
            return {"ui": {"hex_color": []}, "result": ("", "")}

        color_strings = [c.strip() for c in hex_color.split(',') if c.strip()]
        
        found_names = []
        validated_hexes = []

        for color_str in color_strings[:10]:
            hex_proc = color_str.upper()
            if not hex_proc.startswith("#"):
                hex_proc = "#" + hex_proc
            
            if len(hex_proc) == 4:
                hex_proc = f"#{hex_proc[1]*2}{hex_proc[2]*2}{hex_proc[3]*2}"
            
            if not re.match(r'^#[0-9A-F]{6}$', hex_proc):
                continue

            validated_hexes.append(hex_proc)
            name = self._find_closest_color_name(hex_proc, use_css_name)
            found_names.append(name)

        final_names_str = ", ".join(found_names)
        final_hexes_str = ", ".join(validated_hexes)
        
        ui_data = {"hex_color": validated_hexes}
        result_tuple = (final_names_str, final_hexes_str)
        
        return {"ui": ui_data, "result": result_tuple}

# END Hex to Color Name | Sokes 🦬
##############################################################


##############################################################
# START Hex Color Swatch | Sokes 🦬

class hex_color_swatch_sokes:
    CATEGORY = "Sokes 🦬"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex": ("STRING", {"default": "#FF6347, #4682B4, #32CD32", "multiline": False}),
            }
        }

    def execute(self, hex):
        if not hex or not hex.strip():
            return {"ui": {"hex": []}}

        color_strings = [c.strip() for c in hex.split(',') if c.strip()]
        
        processed_hexes = []
        for color_str in color_strings[:10]:
            c = color_str.upper()
            if not c.startswith('#'):
                c = '#' + c
            
            if re.match(r'^#([0-9a-fA-F]{3}){1,2}$', c):
                if len(c) == 4:
                    c = f'#{c[1]*2}{c[2]*2}{c[3]*2}'
                processed_hexes.append(c)

        return {"ui": {"hex": processed_hexes}}

# END Hex Color Swatch | Sokes 🦬
##############################################################


##############################################################
# START Random Number | Sokes 🦬
class random_number_sokes:
    CATEGORY = "Sokes 🦬"; RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN"); RETURN_NAMES = ("integer_output", "float_output", "boolean_output"); FUNCTION = "generate_random_value"
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "minimum": ("FLOAT", {"default": 0.0, "min": -1.0e18, "max": 1.0e18, "step": 0.01}), "maximum": ("FLOAT", {"default": 1.0, "min": -1.0e18, "max": 1.0e18, "step": 0.01}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), } }
    def generate_random_value(self, minimum, maximum, seed):
        min_val, max_val = min(minimum, maximum), max(minimum, maximum)
        random.seed(seed)
        primary_float = random.uniform(min_val, max_val)
        derived_int = int(round(primary_float))
        midpoint = min_val + (max_val - min_val) / 2.0; derived_bool = primary_float > midpoint
        return (derived_int, primary_float, derived_bool)
    @classmethod
    def IS_CHANGED(cls, minimum, maximum, seed):
        h = hashlib.sha256(); h.update(f"min:{minimum}-max:{maximum}-seed:{seed}".encode('utf-8')); return h.hexdigest()

# END Random Number | Sokes 🦬
##############################################################


##############################################################
# START Generate Random Background | Sokes 🦬
class RandomArtGeneratorSokes:
    CATEGORY = "Sokes 🦬/Generators"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "description", "alpha_matte_path")
    FUNCTION = "generate_art"
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory() if folder_paths else "temp_sokes_previews"
        self.type = "temp"
        self.color_name_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        from PIL import ImageFilter, ImageEnhance # Check for imports early
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "bg_type": (["Random", "Solid", "Gradient", "Fog"],),
                "bg_color": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g., #ff0000, 00ff00"}),
                "compositionally_sound": ("BOOLEAN", {"default": True}),
                "alpha_matte_folder": ("STRING", {"default": "", "multiline": False, "placeholder": "Path to folder with PNG/WEBP mattes"}),
                "custom_alpha_color": ("STRING", {"default": "", "multiline": False, "placeholder": "Hex to tint matte, e.g., #FFFFFF"}),
                "minimal_colors": ("BOOLEAN", {"default": False}),
                "num_shapes": ("INT", {"default": 3, "min": 0, "max": 50, "step": 1}),
                "shape_color_mode": (["Any color", "Neutral colors", "Custom colors"],),
                "custom_shapes_colors": ("STRING", {"default": "#FF00FF, #552288, 993322", "multiline": False}),
                "noise_level": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "post_processing": (["Random", "None", "Blur", "Pixelate", "Brightness", "Sharpen", "Vignette", "Chromatic Aberration", "Halftone"],),
                "post_processing_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "primary_color": ("STRING", {"default": "#000000", "placeholder": "Leave blank for random"}),
                "secondary_color": ("STRING", {"default": "#FFFFFF", "placeholder": "Leave blank for random"}),
            },
            "optional": {}
        }

    def _get_color_name(self, hex_color):
        if hex_color in self.color_name_cache: return self.color_name_cache[hex_color]
        if not hex_color or not re.match(r'^#[0-9a-fA-F]{6}$', hex_color): return "unknown color"
        result_name = "color"
        try:
            standard_name = webcolors.hex_to_name(hex_color, spec="css3")
            result_name = human_readable_map.get(standard_name, standard_name)
        except ValueError:
            try:
                requested_rgb = webcolors.hex_to_rgb(hex_color)
                requested_lab = convert_color(sRGBColor(*requested_rgb, is_upscaled=True), LabColor)
                min_dist, closest_name_internal = float('inf'), "color"
                for source_map in [explicit_targets_for_comparison, css3_names_to_hex]:
                    for name, hex_val in source_map.items():
                        try:
                            cand_rgb = webcolors.hex_to_rgb(hex_val)
                            cand_lab = convert_color(sRGBColor(*cand_rgb, is_upscaled=True), LabColor)
                            dist = delta_e_cie2000(requested_lab, cand_lab)
                            if dist < min_dist: min_dist, closest_name_internal = dist, name
                        except Exception: continue
                result_name = human_readable_map.get(closest_name_internal, closest_name_internal)
            except Exception: result_name = "indeterminate color"
        self.color_name_cache[hex_color] = result_name
        return result_name

    def _generate_palette(self, minimal, primary_hex, secondary_hex):
        if primary_hex == "#000000" and secondary_hex == "#FFFFFF":
            if minimal:
                base_r, base_g, base_b = [random.randint(0, 255) for _ in range(3)]
                return [f"#{base_r:02x}{base_g:02x}{base_b:02x}", f"#{max(0, base_r-50):02x}{max(0, base_g-50):02x}{max(0, base_b-50):02x}", f"#{min(255, base_r+50):02x}{min(255, base_g+50):02x}{min(255, base_b+50):02x}"]
            else: 
                return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(random.randint(3, 5))]
        p_color = primary_hex.strip() if primary_hex else f"#{random.randint(0, 0xFFFFFF):06x}"
        s_color = secondary_hex.strip() if secondary_hex else f"#{random.randint(0, 0xFFFFFF):06x}"
        palette = [p_color, s_color]
        if minimal:
            try:
                r, g, b = webcolors.hex_to_rgb(p_color)
                palette.append(f"#{max(0, min(255, r - 40)):02x}{max(0, min(255, g - 40)):02x}{max(0, min(255, b - 40)):02x}")
            except ValueError: pass
        return palette

    def _parse_hex_colors(self, color_string):
        if not color_string: return []
        colors = []
        for color_str in color_string.split(','):
            c = color_str.strip()
            if not c: continue
            if not c.startswith('#'): c = '#' + c
            if re.match(r'^#([0-9a-fA-F]{3}){1,2}$', c):
                if len(c) == 4: c = f'#{c[1]*2}{c[2]*2}{c[3]*2}'
                colors.append(c)
        return colors

    def _adjust_color(self, hex_str, factor):
        r, g, b = webcolors.hex_to_rgb(hex_str)
        r,g,b = int(min(255, max(0, r * factor))), int(min(255, max(0, g * factor))), int(min(255, max(0, b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _apply_halftone(self, image, amount):
        dot_size = int(np.interp(amount, [0, 100], [32, 2]))
        img_gray = image.convert('L')
        img_np = np.array(img_gray)
        new_img = Image.new('RGB', image.size, 'white')
        draw = ImageDraw.Draw(new_img)
        for y in range(0, image.height, dot_size):
            for x in range(0, image.width, dot_size):
                box = img_np[y:y+dot_size, x:x+dot_size]
                if box.size == 0: continue
                brightness = np.mean(box)
                dot_radius = (1 - (brightness / 255)) * (dot_size / 2)
                cx, cy = x + dot_size/2, y + dot_size/2
                draw.ellipse([cx-dot_radius, cy-dot_radius, cx+dot_radius, cy+dot_radius], fill='black')
        return new_img


    def generate_art(self, width, height, seed, bg_type, bg_color, compositionally_sound, alpha_matte_folder, custom_alpha_color, minimal_colors, num_shapes, shape_color_mode, custom_shapes_colors, noise_level, post_processing, post_processing_amount, primary_color, secondary_color):
        from PIL import ImageFilter, ImageEnhance
        random.seed(seed)
        self.color_name_cache.clear()

        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        description_parts, used_alpha_path = [], ""
        
        main_palette = self._generate_palette(minimal_colors, primary_color, secondary_color)
        bg_palette = self._parse_hex_colors(bg_color)
        if not bg_palette: bg_palette = main_palette

        if bg_type == "Random":
            bg_type = random.choice(["Solid", "Gradient", "Fog"])

        if bg_type == "Solid":
            color = random.choice(bg_palette); draw.rectangle([0, 0, width, height], fill=color)
            description_parts.append(f"a solid {self._get_color_name(color)} background")
        elif bg_type == "Gradient":
            c1_hex = random.choice(bg_palette)
            if len(bg_palette) > 1: c2_hex = random.choice([c for c in bg_palette if c != c1_hex] or bg_palette)
            else: c2_hex = self._adjust_color(c1_hex, random.choice([0.5, 1.5]))
            c1_rgb, c2_rgb = webcolors.hex_to_rgb(c1_hex), webcolors.hex_to_rgb(c2_hex); grad_type = random.choice(['linear', 'radial', 'diamond'])
            if grad_type == 'linear':
                axis = random.choice(['x', 'y'])
                if axis == 'y':
                    for y in range(height): r, g, b = [np.interp(y, [0, height], [c1, c2]) for c1,c2 in zip(c1_rgb, c2_rgb)]; draw.line([(0, y), (width, y)], fill=(int(r), int(g), int(b)))
                else:
                     for x in range(width): r, g, b = [np.interp(x, [0, width], [c1, c2]) for c1,c2 in zip(c1_rgb, c2_rgb)]; draw.line([(x, 0), (x, height)], fill=(int(r), int(g), int(b)))
            elif grad_type == 'radial':
                center_x, center_y = random.randint(0, width), random.randint(0, height); max_radius = int(math.sqrt(max(center_x, width-center_x)**2 + max(center_y, height-center_y)**2))
                for r_iter in range(max_radius, 0, -2): r, g, b = [np.interp(r_iter, [0, max_radius], [c2, c1]) for c1,c2 in zip(c1_rgb, c2_rgb)]; draw.ellipse([center_x-r_iter, center_y-r_iter, center_x+r_iter, center_y+r_iter], fill=(int(r), int(g), int(b)))
            elif grad_type == 'diamond':
                center_x, center_y = width / 2, height / 2; max_dist = int((width + height) / 2)
                for d in range(max_dist, 0, -2): r, g, b = [np.interp(d, [0, max_dist], [c2, c1]) for c1,c2 in zip(c1_rgb, c2_rgb)]; points = [(center_x, center_y-d), (center_x+d, center_y), (center_x, center_y+d), (center_x-d, center_y)]; draw.polygon(points, fill=(int(r), int(g), int(b)))
            description_parts.append(f"a {grad_type} gradient from {self._get_color_name(c1_hex)} to {self._get_color_name(c2_hex)}")
        elif bg_type == "Fog":
            c1_hex, c2_hex = random.choice(bg_palette), random.choice(bg_palette)
            if len(bg_palette) == 1: c2_hex = self._adjust_color(c1_hex, random.choice([0.5, 1.5]))
            c1_rgb, c2_rgb = np.array(webcolors.hex_to_rgb(c1_hex)), np.array(webcolors.hex_to_rgb(c2_hex)); noise = np.zeros((height, width))
            for i in range(4):
                divisor = 8 * (2**i); amp = 0.5**(i+1); low_res_w, low_res_h = int(width/divisor), int(height/divisor)
                if low_res_w < 1 or low_res_h < 1: break
                layer = np.random.rand(low_res_h, low_res_w); layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_CUBIC); noise += layer * amp
            noise = (noise - noise.min()) / (noise.max() - noise.min()); noise = np.stack([noise]*3, axis=-1); fog_array = c1_rgb * (1 - noise) + c2_rgb * noise
            image = Image.fromarray(fog_array.astype(np.uint8)); draw = ImageDraw.Draw(image)
            description_parts.append(f"a foggy background of {self._get_color_name(c1_hex)} and {self._get_color_name(c2_hex)}")
        
        if compositionally_sound:
            rule = random.choice(['focal_point', 'split'])
            if rule == 'focal_point':
                thirds_x, thirds_y = [width // 3, width * 2 // 3], [height // 3, height * 2 // 3]; intersections = [(tx, ty) for tx in thirds_x for ty in thirds_y]; pos_names = ["top left", "top right", "bottom left", "bottom right"]
                idx = random.randint(0, 3); point, pos_name = intersections[idx], pos_names[idx]; focal_color_hex, radius = random.choice(main_palette), random.randint(min(width, height) // 10, min(width, height) // 5)
                draw.ellipse([point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius], fill=focal_color_hex); brightness = "bright" if sum(webcolors.hex_to_rgb(focal_color_hex)) > 384 else "dark"
                description_parts.append(f"with a {brightness} {self._get_color_name(focal_color_hex)} spot in the {pos_name} area")
            elif rule == 'split':
                fill_color_hex = random.choice(main_palette)
                if random.random() < 0.2:
                    split_axis = random.choice(['horizontal', 'vertical']); points = []
                    if split_axis == 'horizontal':
                        p1_y, p2_y = random.randint(0, height), random.randint(0, height); side = random.choice(['top', 'bottom'])
                        if side == 'top':
                            points = [(0, p1_y), (width, p2_y), (width, 0), (0, 0)]
                        else:
                            points = [(0, p1_y), (width, p2_y), (width, height), (0, height)]
                    else:
                        p1_x, p2_x = random.randint(0, width), random.randint(0, width); side = random.choice(['left', 'right'])
                        if side == 'left':
                            points = [(p1_x, 0), (p2_x, height), (0, height), (0, 0)]
                        else:
                            points = [(p1_x, 0), (p2_x, height), (width, height), (width, 0)]
                    draw.polygon(points, fill=fill_color_hex); description_parts.append(f"an angled {self._get_color_name(fill_color_hex)} compositional line")
                else:
                    split_type = random.choice(['horizontal', 'vertical'])
                    if split_type == 'horizontal':
                        thirds_y = [height // 3, height * 2 // 3]; line_y = random.choice(thirds_y); desc_pos = "high" if line_y == thirds_y[0] else "low"; y_start, y_end = (0, line_y) if desc_pos == "high" else (line_y, height)
                        draw.rectangle([0, y_start, width, y_end], fill=fill_color_hex); description_parts.append(f"a {desc_pos} {self._get_color_name(fill_color_hex)} horizon line")
                    else:
                        thirds_x = [width // 3, width * 2 // 3]; line_x = random.choice(thirds_x); side = "left" if random.random() > 0.5 else "right"; x_start, x_end = (0, line_x) if side == "left" else (line_x, width)
                        draw.rectangle([x_start, 0, x_end, height], fill=fill_color_hex); description_parts.append(f"a {self._get_color_name(fill_color_hex)} section on the {side} 1/3")
        else: description_parts.append(f"with abstract colors")
        
        matte_files = find_image_files_in_folder(alpha_matte_folder.strip(), False) if alpha_matte_folder and alpha_matte_folder.strip() else []
        if matte_files:
            used_alpha_path = random.choice(matte_files)
            try:
                matte_img = Image.open(used_alpha_path).convert("RGBA").resize((width, height), Image.LANCZOS)
                clean_color = custom_alpha_color.strip()
                if clean_color and re.match(r'^#?([0-9a-fA-F]{3}){1,2}$', clean_color):
                    if not clean_color.startswith("#"): clean_color = "#" + clean_color
                    color_layer = Image.new("RGBA", (width, height), color=clean_color); alpha_mask = matte_img.getchannel('A')
                    image.paste(color_layer, (0, 0), alpha_mask); description_parts.append(f"overlaid with a {self._get_color_name(clean_color)} matte")
                else:
                    image.paste(matte_img, (0,0), matte_img); description_parts.append(f"overlaid with a matte")
            except Exception as e: print(f"sokes_nodes.py: Could not process alpha matte '{os.path.basename(used_alpha_path)}': {e}"); used_alpha_path = ""
        
        shape_palette = [];
        if shape_color_mode == "Neutral colors": shape_palette = ['#000000', '#FFFFFF', '#808080', '#A9A9A9', '#D3D3D3', '#2F4F4F', '#696969', '#A52A2A', '#8B4513', '#D2B48C']
        elif shape_color_mode == "Custom colors": shape_palette = self._parse_hex_colors(custom_shapes_colors)
        if not shape_palette and shape_color_mode != "Any color": shape_palette = main_palette
        remaining_shapes = num_shapes
        if remaining_shapes > 0 and random.random() < 0.1: self._draw_random_shape(image, width, height, True, shape_color_mode, shape_palette); remaining_shapes -= 1
        for _ in range(remaining_shapes): self._draw_random_shape(image, width, height, False, shape_color_mode, shape_palette)
        if num_shapes > 0:
            desc_part = f"and scattered with {num_shapes} random shapes"
            if shape_color_mode == "Neutral colors": desc_part += " in neutral tones"
            elif shape_color_mode == "Custom colors": desc_part += " in custom colors"
            description_parts.append(desc_part)
        
        if noise_level > 0:
            np_image = np.array(image).astype(np.float32); noise_intensity = noise_level * 7.5
            noise = np.random.normal(0, noise_intensity, np_image.shape); np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8); image = Image.fromarray(np_image)

        if post_processing == "Random":
            post_processing = random.choice(["None", "Blur", "Pixelate", "Brightness", "Sharpen", "Vignette", "Chromatic Aberration", "Halftone"])

        if post_processing != "None" and post_processing_amount > 0:
            amount = post_processing_amount
            if post_processing == "Blur":
                blur_radius = np.interp(amount, [0, 100], [0, 20]); image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            elif post_processing == "Pixelate":
                pixel_size = int(np.interp(amount, [0, 100], [1, 64]))
                if pixel_size > 1: img_small = image.resize((width//pixel_size, height//pixel_size), resample=Image.NEAREST); image = img_small.resize(image.size, resample=Image.NEAREST)
            elif post_processing == "Brightness":
                factor = np.interp(amount, [0, 100], [0.5, 1.5]); enhancer = ImageEnhance.Brightness(image); image = enhancer.enhance(factor)
            elif post_processing == "Sharpen":
                percent_val = int(np.interp(amount, [0,100],[100, 300])); image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent_val, threshold=3))
            elif post_processing == "Chromatic Aberration":
                shift = int(np.interp(amount, [0, 100], [0, 20])); r, g, b = image.split()
                r = r.transform(r.size, Image.AFFINE, (1, 0, shift, 0, 1, 0)); b = b.transform(b.size, Image.AFFINE, (1, 0, -shift, 0, 1, 0)); image = Image.merge("RGB", (r, g, b))
            elif post_processing == "Vignette":
                strength = np.interp(amount, [0, 100], [0, 1])
                x_ax = np.linspace(-1, 1, width); y_ax = np.linspace(-1, 1, height); xx, yy = np.meshgrid(x_ax, y_ax)
                radius = np.sqrt(xx**2 + yy**2); vignette_mask = 1 - np.clip(radius * strength, 0, 1)
                img_np = np.array(image) * vignette_mask[:, :, np.newaxis]; image = Image.fromarray(img_np.astype(np.uint8))
            elif post_processing == "Halftone":
                image = self._apply_halftone(image, amount)
            description_parts.append(f"and a {post_processing.lower()} effect")
        
        image_np = np.array(image).astype(np.float32) / 255.0; image_tensor = torch.from_numpy(image_np)[None,]
        final_description = ". ".join(description_parts).capitalize() + "."
        
        previews_out = []
        if preview_available:
            preview_subfolder = "sokes_art_previews"; full_output_folder = os.path.join(self.output_dir, preview_subfolder)
            if not os.path.exists(full_output_folder): os.makedirs(full_output_folder)
            preview_filename = f"art_{seed}_{hashlib.sha1(final_description.encode()).hexdigest()[:6]}.png"
            filepath = os.path.join(full_output_folder, preview_filename); image.save(filepath, compress_level=4)
            previews_out.append({"filename": preview_filename, "subfolder": preview_subfolder, "type": self.type})

        return {"ui": {"images": previews_out}, "result": (image_tensor, final_description, used_alpha_path)}

    def _draw_random_shape(self, image, width, height, is_giant, shape_color_mode, palette):
        if is_giant: shape_w, shape_h = random.randint(int(width*0.7), int(width*0.9)), random.randint(int(height*0.7), int(height*0.9))
        else: shape_w, shape_h = random.randint(width//30, width//3), random.randint(height//30, height//3)
        if shape_color_mode == "Any color": shape_color = f"#{random.randint(0, 0xFFFFFF):06x}"
        else: shape_color = random.choice(palette) if palette else f"#{random.randint(0, 0xFFFFFF):06x}"
        shape_type = random.choice(['circle', 'triangle', 'rectangle', 'star', 'diamond', 'cross', 'pentagon', 'hexagon'])
        if shape_type == 'circle':
            x1, y1 = random.randint(-shape_w//2, width - shape_w//2), random.randint(-shape_h//2, height - shape_h//2)
            ImageDraw.Draw(image).ellipse([x1, y1, x1+shape_w, y1+shape_h], fill=shape_color)
            return
        max_dim = int(max(shape_w, shape_h) * 1.5); temp_img = Image.new('RGBA', (max_dim, max_dim))
        temp_draw = ImageDraw.Draw(temp_img); cx, cy = max_dim // 2, max_dim // 2
        points = []
        if shape_type == 'rectangle': temp_draw.rectangle([cx-shape_w//2, cy-shape_h//2, cx+shape_w//2, cy+shape_h//2], fill=shape_color)
        elif shape_type == 'cross':
            arm_w, arm_h = shape_w // 3, shape_h // 3
            temp_draw.rectangle([cx-shape_w//2, cy-arm_h//2, cx+shape_w//2, cy+arm_h//2], fill=shape_color)
            temp_draw.rectangle([cx-arm_w//2, cy-shape_h//2, cx+arm_w//2, cy+shape_h//2], fill=shape_color)
        else:
            num_points = {'triangle': 3, 'diamond': 4, 'pentagon': 5, 'hexagon': 6}.get(shape_type)
            if num_points:
                angle_step = 360 / num_points
                for i in range(num_points): points.append((cx + shape_w/2 * math.cos(math.radians(i*angle_step-90)), cy + shape_h/2 * math.sin(math.radians(i*angle_step-90))))
            elif shape_type == 'star':
                outer_r, inner_r = shape_w/2, shape_h/4
                for i in range(5):
                    points.append((cx + outer_r * math.cos(math.radians(i*72-90)), cy + outer_r * math.sin(math.radians(i*72-90))))
                    points.append((cx + inner_r * math.cos(math.radians(i*72+36-90)), cy + inner_r * math.sin(math.radians(i*72+36-90))))
            temp_draw.polygon(points, fill=shape_color)
        rotated_shape = temp_img.rotate(random.randint(0, 359), expand=True, resample=Image.BICUBIC)
        paste_x, paste_y = random.randint(-rotated_shape.width//2, width - rotated_shape.width//2), random.randint(-rotated_shape.height//2, height - rotated_shape.height//2)
        image.paste(rotated_shape, (paste_x, paste_y), rotated_shape)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return hashlib.sha256(str(kwargs).encode('utf-8')).hexdigest()

# END Generate Random Background | Sokes 🦬
##############################################################


##############################################################
# START Random Hex Color | Sokes 🦬

class RandomHexColorSokes:
    CATEGORY = "Sokes 🦬/Generators"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex_color_string",)
    FUNCTION = "generate_hex_colors"

    COLOR_TYPES = [
        "random", "normal", "neon", "pale", "muted", "dark colors",
        "warm colors", "cool colors", "tan colors", "light grays", 
        "dark grays", "very dark grays", "all grays"
    ]
    
    NORMAL_COLORS = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FFA500', '#800080',
        '#FFC0CB', '#A52A2A', '#FFFFFF', '#000000', '#808080'
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "color_type": (cls.COLOR_TYPES, {"default": "random"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        if s == 0.0:
            v_int = int(v * 255)
            return (v_int, v_int, v_int)
        
        c = v * s
        h_prime = h / 60.0
        x = c * (1 - abs(h_prime % 2 - 1))
        
        r1, g1, b1 = 0.0, 0.0, 0.0
        
        if 0 <= h_prime < 1:
            r1, g1, b1 = c, x, 0
        elif 1 <= h_prime < 2:
            r1, g1, b1 = x, c, 0
        elif 2 <= h_prime < 3:
            r1, g1, b1 = 0, c, x
        elif 3 <= h_prime < 4:
            r1, g1, b1 = 0, x, c
        elif 4 <= h_prime < 5:
            r1, g1, b1 = x, 0, c
        elif 5 <= h_prime <= 6:
            r1, g1, b1 = c, 0, x
            
        m = v - c
        return (int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255))

    def _generate_one_hex(self, color_type):
        if color_type == "random": return f"#{random.randint(0, 0xFFFFFF):06x}"
        elif color_type == "normal": return random.choice(self.NORMAL_COLORS)
        elif color_type == "neon": r, g, b = self._hsv_to_rgb(random.uniform(0, 360), random.uniform(0.9, 1.0), random.uniform(0.95, 1.0))
        elif color_type == "pale": r, g, b = self._hsv_to_rgb(random.uniform(0, 360), random.uniform(0.1, 0.25), random.uniform(0.9, 1.0))
        elif color_type == "muted": r, g, b = self._hsv_to_rgb(random.uniform(0, 360), random.uniform(0.15, 0.35), random.uniform(0.4, 0.7))
        elif color_type == "dark colors": r, g, b = self._hsv_to_rgb(random.uniform(0, 360), random.uniform(0.7, 1.0), random.uniform(0.07, 0.4))
        elif color_type == "warm colors": r, g, b = self._hsv_to_rgb((random.uniform(-81, 59) + 360) % 360, random.uniform(0.65, 1.0), random.uniform(0.5, 1.0))
        elif color_type == "cool colors": r, g, b = self._hsv_to_rgb(random.uniform(120, 240), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0))
        elif color_type == "tan colors": r, g, b = self._hsv_to_rgb(random.uniform(36, 50), random.uniform(0.02, 0.16), random.uniform(0.8, 1.0))
        elif color_type == "light grays": gray_val = random.randint(192, 224); r, g, b = gray_val, gray_val, gray_val
        elif color_type == "dark grays": gray_val = random.randint(64, 96); r, g, b = gray_val, gray_val, gray_val
        elif color_type == "very dark grays": gray_val = random.randint(2, 48); r, g, b = gray_val, gray_val, gray_val
        elif color_type == "all grays": gray_val = random.randint(0, 255); r, g, b = gray_val, gray_val, gray_val
        else: return f"#{random.randint(0, 0xFFFFFF):06x}"
        return f"#{r:02x}{g:02x}{b:02x}"

    def generate_hex_colors(self, count, color_type, seed):
        random.seed(seed)
        hex_colors = [self._generate_one_hex(color_type).upper() for _ in range(count)]
        final_string = ", ".join(hex_colors)
        return {"ui": {"hex_color_string": [final_string]}, "result": (final_string,)}

    @classmethod
    def IS_CHANGED(cls, count, color_type, seed):
        return hashlib.sha256(f"{count}-{color_type}-{seed}".encode('utf-8')).hexdigest()

# END Random Hex Color | Sokes 🦬
##############################################################


##############################################################
# START Street View Loader | Sokes 🦬

class StreetViewLoaderSokes:
    """
    Loads Google Street View images.
    
    Requires environment variable: GOOGLE_STREET_VIEW_API_KEY
    Get your API key from: https://console.cloud.google.com/marketplace/product/google/street-view-image-backend.googleapis.com
    """
    RESOLUTIONS = ["640x640 (1:1)", "640x480 (4:3)", "512x512 (1:1)"]
    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_street_view_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.RESOLUTIONS,),
                "location_mode": (["Address", "Latitude/Longitude"],),
                "address": ("STRING", {"multiline": False, "default": "Eiffel Tower, Paris, France"}),
                "latitude": ("FLOAT", {"default": 48.85826, "min": -90.0, "max": 90.0, "step": 0.0001}),
                "longitude": ("FLOAT", {"default": 2.2945, "min": -180.0, "max": 180.0, "step": 0.0001}),
                "fov": ("INT", {"default": 90, "min": 10, "max": 120, "step": 1}),
                "heading": ("FLOAT", {"default": 235.0, "min": 0, "max": 360, "step": 0.001}),
                "pitch": ("FLOAT", {"default": 10.0, "min": -90, "max": 90, "step": 0.001}),
                "google_source": (["default", "outdoor"],),
            },
            "optional": {
                "point_at_address": ("BOOLEAN", {"default": False}),
            }
        }

    def get_coords_from_address(self, address):
        encoded_address = urllib.parse.quote(address)
        # Using a public, reliable geocoding service
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json"
        headers = {"User-Agent": "ComfyUI-Sokes-Nodes/1.0"}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
            else:
                print(f"StreetViewLoader WARNING: Geocoding found no results for address: {address}")
                return None, None
        except Exception as e:
            print(f"StreetViewLoader ERROR: Geocoding request failed: {e}")
            return None, None

    def calculate_heading(self, point_a, point_b):
        lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
        lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])
        dLon = lon2 - lon1
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def get_view_parameters_for_target(self, target_lat, target_lon):
        metadata_url = (f"https://maps.googleapis.com/maps/api/streetview/metadata"
                        f"?location={target_lat},{target_lon}&key={GOOGLE_STREET_VIEW_API_KEY}")
        try:
            r = requests.get(metadata_url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get('status') == 'OK':
                pano_location = data['location']
                pano_lat, pano_lon = pano_location['lat'], pano_location['lng']
                heading = self.calculate_heading((pano_lat, pano_lon), (target_lat, target_lon))
                return {"heading": heading, "latitude": pano_lat, "longitude": pano_lon}
            else:
                error_msg = data.get('error_message', f"Could not find panorama for {target_lat},{target_lon}.")
                print(f"StreetViewLoader WARNING: Metadata lookup failed ({data.get('status')}): {error_msg}")
                return None
        except Exception as e:
            print(f"StreetViewLoader ERROR: Failed to fetch metadata for target {target_lat},{target_lon}: {e}")
            return None

    def load_street_view_image(self, resolution, location_mode, address, latitude, longitude, fov, heading, pitch, google_source, point_at_address=False):
        res_string = resolution.split(' ')[0]
        width, height = map(int, res_string.split('x'))
        
        # Check for API key first
        if not GOOGLE_STREET_VIEW_API_KEY:
            print("StreetViewLoader ERROR: Google API Key is missing. Please set environment variable GOOGLE_STREET_VIEW_API_KEY.")
            # Create an error image with text
            error_img = Image.new('RGB', (width, height), color='#FF6B6B')
            draw = ImageDraw.Draw(error_img)
            
            # Try to use a default font, fallback to basic text if font not available
            try:
                from PIL import ImageFont
                # Try to use a system font
                font_size = max(12, min(width, height) // 20)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            except ImportError:
                font = None
            
            error_text = "Add API key\nGOOGLE_STREET_VIEW_API_KEY\nin your environment variables"
            text_lines = error_text.split('\n')
            
            # Calculate text position (center)
            if font:
                # Get text size for centering
                bbox = draw.textbbox((0, 0), error_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                # Draw text in white
                draw.text((x, y), error_text, font=font, fill='#FFFFFF')
            else:
                # Fallback to basic text positioning
                x = width // 4
                y = height // 3
                for i, line in enumerate(text_lines):
                    draw.text((x, y + i * 20), line, fill='#FFFFFF')
            
            # Convert to tensor
            error_np = np.array(error_img).astype(np.float32) / 255.0
            return (torch.from_numpy(error_np)[None,],)
        
        error_tensor = (torch.zeros(1, height, width, 3, dtype=torch.float32),)

        loc_lat, loc_lon = latitude, longitude
        if location_mode == "Address":
            lat, lon = self.get_coords_from_address(address)
            if lat is not None:
                loc_lat, loc_lon = lat, lon
            else:
                return error_tensor

        final_params = {"heading": heading, "latitude": loc_lat, "longitude": loc_lon, "pitch": pitch}

        if point_at_address:
            print(f"StreetViewLoader: Auto-pointing enabled. Looking for panorama near target: {loc_lat},{loc_lon}...")
            view_params = self.get_view_parameters_for_target(loc_lat, loc_lon)
            if view_params:
                final_params = view_params
                # Reset pitch to 0 when auto-pointing at address for better viewing
                final_params["pitch"] = 0.0
                print(f"StreetViewLoader: Found panorama. Using calculated heading: {final_params['heading']:.2f}° from {final_params['latitude']},{final_params['longitude']} with pitch reset to 0°")
            else:
                print("StreetViewLoader WARNING: Auto-pointing failed. Falling back to manual settings and original coordinates.")

        api_url = (f"https://maps.googleapis.com/maps/api/streetview"
                   f"?size={width}x{height}&location={final_params['latitude']},{final_params['longitude']}"
                   f"&heading={final_params['heading']}&pitch={final_params['pitch']}&fov={fov}&source={google_source}"
                   f"&return_error_code=true&key={GOOGLE_STREET_VIEW_API_KEY}")
        
        try:
            r = requests.get(api_url, timeout=15)
            if 'Sorry, we have no imagery here.' in r.text or r.status_code == 404:
                 print("StreetViewLoader ERROR: No image found for this location.")
                 return error_tensor
            r.raise_for_status()
            
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return (torch.from_numpy(img_np)[None,],)
        except requests.exceptions.RequestException as e:
            print(f"StreetViewLoader ERROR: Network/API request failed: {e}")
            return error_tensor
        except Exception as e:
            print(f"StreetViewLoader ERROR: Failed to process image: {e}")
            return error_tensor

# END Street View Loader | Sokes 🦬
##############################################################


##############################################################
# START Runpod Serverless | Sokes 🦬

class RunpodServerlessSokes:
    """
    Integrates with Runpod Serverless API.
    
    Requires environment variable: RUNPOD_API_KEY
    Get your API key from: https://runpod.io/console/user/settings
    """
    CATEGORY = "Sokes 🦬/Integrations"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("response_text", "full_response", "status", "execution_time")
    FUNCTION = "call_runpod_endpoint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {
                    "default": "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run",
                    "multiline": False,
                    "placeholder": "https://api.runpod.ai/v2/pz1ury0jpm654q/run"
                }),
                "model": ("STRING", {
                    "default": "llava:7b",
                    "multiline": False,
                    "placeholder": "Vision: llava:7b | Text-only: gemma3:27b"
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 3600,
                    "step": 10,
                    "display": "number"
                }),
            },
            "optional": {
                "prompt_text": ("STRING", {
                    "default": "Describe this image in detail",
                    "multiline": True,
                    "placeholder": "Text prompt for the model"
                }),
                "image": ("IMAGE", {
                    "placeholder": "Optional: Connect image input for visual language models"
                }),
            }
        }

    def call_runpod_endpoint(self, endpoint_url: str, model: str, timeout: int, prompt_text: str = "Describe this image in detail", image=None) -> Tuple[str, str, str, float]:
        start_time = time.time()
        try:
            final_api_key = os.getenv('RUNPOD_API_KEY', '')
            if not final_api_key:
                error_msg = (
                    "No Runpod API key found. Set environment variable RUNPOD_API_KEY."
                )
                print(f"sokes_nodes.py Runpod: {error_msg}")
                return error_msg, error_msg, "NO_API_KEY", 0.0

            payload_dict = {
                "input": {
                    "model": model.strip(),
                    "prompt": prompt_text.strip() if prompt_text.strip() else "Describe this image in detail",
                    "stream": False
                }
            }

            if image is not None:
                print("sokes_nodes.py Runpod: Converting image to base64…")
                image_b64 = self._convert_image_to_base64(image)
                if image_b64:
                    payload_dict["input"]["image"] = image_b64
                else:
                    print("sokes_nodes.py Runpod: Failed to convert image to base64.")
            else:
                print("sokes_nodes.py Runpod: No image provided (text-only request).")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {final_api_key}"
            }

            print(f"sokes_nodes.py Runpod: Calling endpoint {endpoint_url}")

            import copy
            log_payload = copy.deepcopy(payload_dict)
            if "input" in log_payload and "image" in log_payload["input"]:
                image_data = payload_dict["input"]["image"]
                log_payload["input"]["image"] = f"[IMAGE_DATA: {len(image_data)} chars]"
            print(f"sokes_nodes.py Runpod: Payload = {json.dumps(log_payload, indent=2)}")

            response = requests.post(
                endpoint_url.strip(),
                json=payload_dict,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                error_msg = f"Runpod API error: {response.status_code} - {response.text}"
                print(f"sokes_nodes.py Runpod: {error_msg}")
                return error_msg, error_msg, "ERROR", time.time() - start_time

            initial_response = response.json()
            job_id = initial_response.get("id")
            if not job_id:
                error_msg = f"No job ID returned from Runpod: {initial_response}"
                print(f"sokes_nodes.py Runpod: {error_msg}")
                return error_msg, error_msg, "ERROR", time.time() - start_time

            final_response = self._poll_job_completion(job_id, final_api_key, endpoint_url.strip(), timeout, start_time)
            if final_response["status"] == "ERROR":
                return (
                    final_response["error"],
                    final_response["error"],
                    "ERROR",
                    final_response["execution_time"]
                )

            response_text = self._extract_response_text(final_response["data"])
            execution_time = time.time() - start_time
            return (
                response_text,
                json.dumps(final_response["data"], indent=2),
                "SUCCESS",
                execution_time
            )
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout} seconds"
            print(f"sokes_nodes.py RunPod: {error_msg}")
            return error_msg, error_msg, "TIMEOUT", time.time() - start_time
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            print(f"sokes_nodes.py RunPod: {error_msg}")
            return error_msg, error_msg, "ERROR", time.time() - start_time
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"sokes_nodes.py RunPod: {error_msg}")
            return error_msg, error_msg, "ERROR", time.time() - start_time

    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        if not isinstance(response_data, dict):
            text = str(response_data)
            return text.rstrip('\n\r ')
        for field in ["generated_text", "response", "text", "result", "content"]:
            if field in response_data and response_data[field]:
                text = str(response_data[field])
                return text.rstrip('\n\r ')
        raw_response = response_data.get("raw_response", {})
        if isinstance(raw_response, dict) and "response" in raw_response:
            text = str(raw_response["response"]) 
            return text.rstrip('\n\r ')
        output = response_data.get("output", {})
        if isinstance(output, str):
            return output.rstrip('\n\r ')
        if isinstance(output, dict):
            for field in ["response", "generated_text", "text", "result", "content"]:
                if field in output and output[field]:
                    text = str(output[field])
                    return text.rstrip('\n\r ')
            return json.dumps(output, indent=2)
        return json.dumps(response_data, indent=2)

    def _convert_image_to_base64(self, image_tensor) -> str:
        try:
            if image_tensor is None:
                return ""
            if hasattr(image_tensor, 'shape') and len(image_tensor.shape) == 4:
                image_array = image_tensor[0]
            else:
                image_array = image_tensor
            if hasattr(image_array, 'cpu'):
                image_array = image_array.cpu().numpy()
            if not isinstance(image_array, np.ndarray):
                image_array = np.array(image_array)
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                pil_image = Image.fromarray(image_array, 'RGB')
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"sokes_nodes.py Runpod: Error converting image to base64: {e}")
            return ""

    def _poll_job_completion(self, job_id: str, api_key: str, endpoint_url: str, timeout: int, start_time: float) -> Dict[str, Any]:
        try:
            url_parts = endpoint_url.rstrip('/').split('/')
            if len(url_parts) >= 2 and url_parts[-1] == 'run':
                endpoint_id = url_parts[-2]
                status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
            else:
                status_url = f"https://api.runpod.ai/v2/status/{job_id}"
        except Exception as e:
            print(f"sokes_nodes.py Runpod: Error parsing endpoint URL, using fallback: {e}")
            status_url = f"https://api.runpod.ai/v2/status/{job_id}"

        headers = {"Authorization": f"Bearer {api_key}"}
        poll_interval = 2
        max_poll_interval = 10
        poll_count = 0

        while time.time() - start_time < timeout:
            poll_count += 1
            elapsed = time.time() - start_time
            try:
                print(f"sokes_nodes.py Runpod: Poll #{poll_count} ({elapsed:.1f}s)")
                response = requests.get(status_url, headers=headers, timeout=30)
                if response.status_code != 200:
                    print(f"sokes_nodes.py Runpod: Status check failed: {response.status_code} - {response.text}")
                    time.sleep(poll_interval)
                    continue
                status_data = response.json()
                job_status = status_data.get("status", "UNKNOWN")
                if job_status == "COMPLETED":
                    output = status_data.get("output", {})
                    return {"status": "SUCCESS", "data": output, "execution_time": time.time() - start_time}
                elif job_status == "FAILED":
                    error_msg = status_data.get("error", "Job failed with no error message")
                    return {"status": "ERROR", "error": f"Runpod job failed: {error_msg}", "execution_time": time.time() - start_time}
                elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                    time.sleep(poll_interval)
                    if poll_count % 5 == 0:
                        poll_interval = min(poll_interval + 1, max_poll_interval)
                else:
                    time.sleep(poll_interval)
            except requests.exceptions.RequestException as e:
                print(f"sokes_nodes.py RunpPod: Error polling status: {e}")
                time.sleep(poll_interval)
                continue

        return {"status": "ERROR", "error": f"Job polling timeout after {timeout} seconds", "execution_time": time.time() - start_time}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Change hash on inputs and API key suffix; helps ComfyUI cache invalidation
        api_key = os.getenv('RUNPOD_API_KEY', '')
        key_suffix = api_key[-6:] if api_key else "no_key"
        h = hashlib.sha256()
        h.update(str(args).encode('utf-8'))
        h.update(str(kwargs).encode('utf-8'))
        h.update(key_suffix.encode('utf-8'))
        return h.hexdigest()

# END Runpod Serverless | Sokes 🦬
##############################################################


##############################################################
# Node Mappings

NODE_CLASS_MAPPINGS = {
    "Image Picker | sokes 🦬": ImagePickerSokes,
    "Current Date & Time | sokes 🦬": current_date_time_sokes,
    "Latent Switch x9 | sokes 🦬": latent_input_switch_9x_sokes,
    "Replace Text with RegEx | sokes 🦬": replace_text_regex_sokes,
    "Load Random Image | sokes 🦬": load_random_image_sokes,
    "ComfyUI Folder Paths | sokes 🦬": ComfyUI_folder_paths_sokes,
    "Hex to Color Name | sokes 🦬": hex_to_color_name_sokes,
    "Hex Color Swatch | sokes 🦬": hex_color_swatch_sokes,
    "Random Number | sokes 🦬": random_number_sokes,
    "Generate Random Background | sokes 🦬": RandomArtGeneratorSokes,
    "Random Hex Color | sokes 🦬": RandomHexColorSokes,
    "Street View Loader | sokes 🦬": StreetViewLoaderSokes,
    "Runpod Serverless | sokes 🦬": RunpodServerlessSokes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Picker | sokes 🦬": "Image Picker 🦬",
    "Current Date & Time | sokes 🦬": "Current Date & Time 🦬",
    "Latent Switch x9 | sokes 🦬": "Latent Switch x9 🦬",
    "Replace Text with RegEx | sokes 🦬": "Replace Text with RegEx 🦬",
    "Load Random Image | sokes 🦬": "Load Random Image 🦬",
    "ComfyUI Folder Paths | sokes 🦬": "ComfyUI Folder Paths 🦬",
    "Hex to Color Name | sokes 🦬": "Hex to Color Name 🦬",
    "Hex Color Swatch | sokes 🦬": "Hex Color Swatch 🦬",
    "Random Number | sokes 🦬": "Random Number 🦬",
    "Generate Random Background | sokes 🦬": "Generate Random Background 🦬",
    "Random Hex Color | sokes 🦬": "Random Hex Color 🦬",
    "Street View Loader | sokes 🦬": "Street View Loader 🦬",
    "Runpod Serverless | sokes 🦬": "Runpod Serverless 🦬",
}

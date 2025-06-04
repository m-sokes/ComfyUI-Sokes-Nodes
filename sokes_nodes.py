import sys # For sys.path debugging if needed in the future
import os
from datetime import datetime # datetime
import re # regex
import random # for load random image
import hashlib # for random number
import imghdr # to check image by header vs extension
import torch # to resize images
import numpy as np # for image manipulation in torch
import cv2 # for image processing
from PIL import Image, ImageOps # load random image, orient image

import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

from .sokes_color_maps import css3_names_to_hex, css3_hex_to_names, human_readable_map, explicit_targets_for_comparison

if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: a.item()

# --- ComfyUI Integration Imports & Preview Logic ---
# Try importing folder_paths first as it's used by the node's __init__
try:
    import folder_paths
    # print("sokes_nodes.py: Successfully imported 'folder_paths'.")
except ImportError:
    # print("sokes_nodes.py: 'folder_paths' module not found. Path resolution might be limited.")
    folder_paths = None # Set to None if not available

preview_available = False # Assume previews are NOT available by default
PromptServer = None
# FONT_PATH is not strictly needed by this node for previews, but we check for comfy.utils
nodes_module = None # Renamed to avoid conflict with class name 'nodes' if any
comfy_utils_available = False

#print("sokes_nodes.py: Attempting to import ComfyUI specific modules for previews...")
try:
    from server import PromptServer
    #print("sokes_nodes.py:  - Successfully imported 'PromptServer' from 'server'.")
except ImportError as e:
    print(f"sokes_nodes.py:  - FAILED to import 'PromptServer' from 'server'. Error: {e}")
except Exception as e:
    print(f"sokes_nodes.py:  - FAILED to import 'PromptServer' from 'server' with OTHER error: {e}")

try:
    import comfy.utils
    comfy_utils_available = True
    #print("sokes_nodes.py:  - Successfully imported 'comfy.utils' module.")
    #try:
    #    from comfy.utils import FONT_PATH # This specific import might fail, but it's not critical
        #print("sokes_nodes.py:    - Successfully imported 'FONT_PATH' from 'comfy.utils'.")
    #except ImportError:
    #    print("sokes_nodes.py:    - Note: 'FONT_PATH' not found in 'comfy.utils' (this is okay for previews).")
except ImportError as e:
    print(f"sokes_nodes.py:  - FAILED to import 'comfy.utils' module. Error: {e}")
except Exception as e:
    print(f"sokes_nodes.py:  - FAILED to import 'comfy.utils' module with OTHER error: {e}")

try:
    import nodes # This is ComfyUI's 'nodes.py'
    nodes_module = nodes
    #print("sokes_nodes.py:  - Successfully imported ComfyUI 'nodes' module.")
except ImportError as e:
    print(f"sokes_nodes.py:  - FAILED to import ComfyUI 'nodes' module. Error: {e}")
except Exception as e:
    print(f"sokes_nodes.py:  - FAILED to import ComfyUI 'nodes' module with OTHER error: {e}")

if PromptServer and comfy_utils_available and nodes_module:
    try:
        PromptServer.instance # Crucial check: is the server actually running and instance available?
        preview_available = True
        #print("sokes_nodes.py: Preview system checks passed. Previews should be available.")
    except AttributeError:
        #print("sokes_nodes.py: 'PromptServer.instance' not found (AttributeError). Previews will be disabled.")
        preview_available = False
    except Exception as e:
        #print(f"sokes_nodes.py: 'PromptServer.instance' check failed ({type(e).__name__}: {e}). Previews will be disabled.")
        preview_available = False
else:
    missing_components_msg = []
    if not PromptServer: missing_components_msg.append("'PromptServer'")
    if not comfy_utils_available: missing_components_msg.append("'comfy.utils' module")
    if not nodes_module: missing_components_msg.append("ComfyUI 'nodes' module")
    if missing_components_msg:
        print(f"sokes_nodes.py: One or more ComfyUI components for previews failed to import ({', '.join(missing_components_msg)}). Previews disabled.")
    # If components imported but PromptServer.instance failed, that message was already printed.

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
    OUTPUT_NODE = True # This was True, assuming it's an output node, kept it.
                       # If it's just a switch, it might not need to be an OUTPUT_NODE.
    CATEGORY = "Sokes 🦬"

    def latent_input_switch_9x_sokes(self, latent_select, latent_0, latent_1=None, latent_2=None, latent_3=None, latent_4=None, latent_5=None, latent_6=None, latent_7=None, latent_8=None):
        selected_latent = None
        latent_map = {
            0: latent_0, 1: latent_1, 2: latent_2, 3: latent_3, 4: latent_4,
            5: latent_5, 6: latent_6, 7: latent_7, 8: latent_8
        }
        
        select_idx = int(round(latent_select))
        
        if select_idx in latent_map and latent_map[select_idx] is not None:
            selected_latent = latent_map[select_idx]
        else:
            # Fallback to latent_0 if selection is invalid or selected latent is None
            # (and latent_0 itself is not None, though it's required)
            selected_latent = latent_0
            if select_idx not in latent_map:
                 print(f"latent_input_switch_9x_sokes: Invalid latent_select index {select_idx}. Defaulting to latent_0.")
            elif latent_map[select_idx] is None and select_idx != 0 : # only print if not trying to select a None optional input
                 print(f"latent_input_switch_9x_sokes: latent_{select_idx} is None. Defaulting to latent_0.")


        return (selected_latent,)

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

    #@ staticmethod # Original code had this, but it's not strictly necessary for ComfyUI nodes
    def fn_replace_text_regex_sokes(self, regex_pattern, new, text): # Added self
        return (re.sub(regex_pattern, new, text),)


# END Replace Text with Regex | Sokes 🦬
##############################################################

##############################################################
# START Load Random Image with Path and Mask | Sokes 🦬

class load_random_image_sokes:
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        # Use the globally imported folder_paths if available
        self.output_dir = folder_paths.get_temp_directory() if folder_paths else "temp_sokes_previews"
        self.type = "temp" # ComfyUI expects "temp" for previews saved in temp dir
        
        # Create directory if folder_paths was None and we're using a local fallback
        if not folder_paths and not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                print(f"sokes_nodes.py load_random_image_sokes: Warning: Could not create temp directory {self.output_dir}: {e}")
        elif folder_paths and self.type == "temp" and not os.path.exists(self.output_dir):
            # This case should ideally be handled by ComfyUI itself ensuring temp dir exists
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                 print(f"sokes_nodes.py load_random_image_sokes: Warning: Could not create ComfyUI temp directory {self.output_dir}: {e}")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ".", "multiline": False}),
                "search_subfolders": ("BOOLEAN", {"default": False}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sort": ("BOOLEAN", {"default": False}),
                "export_with_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST") # LIST for image_path
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image_or_file"

    OUTPUT_NODE = True # This node produces data and has previews, so True is appropriate.

    def find_image_files(self, folder_path, search_subfolders):
        image_paths = []
        img_extensions_lower = [ext.lower() for ext in self.IMG_EXTENSIONS]
        try:
            if search_subfolders:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        if os.path.isfile(full_path) and file.lower().endswith(tuple(img_extensions_lower)):
                           image_paths.append(full_path)
            else:
                # Check if folder_path itself is valid before listing
                if not os.path.isdir(folder_path):
                    raise FileNotFoundError(f"Directory not found for listing: {folder_path}")
                all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(tuple(img_extensions_lower))]
        except FileNotFoundError as fnf_e: # Catch specific error
             raise fnf_e # Re-raise to be handled by caller
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder_path}': {e}")
        return image_paths

    def validate_and_load_image(self, image_path, final_image_mode):
        try:
            img_pil = Image.open(image_path)
            img_pil = ImageOps.exif_transpose(img_pil) # Orient image based on EXIF

            # Validate image content using imghdr first
            actual_format = imghdr.what(image_path)
            if not actual_format:
                # Fallback to PIL verification if imghdr fails but file might be valid PIL image
                try:
                    img_pil.verify() # This will raise an exception on corrupt images
                except Exception as pil_e:
                    raise ValueError(f"Invalid or corrupt image file (PIL verify failed after imghdr): {image_path} - {str(pil_e)}")
            
            # Convert to the target mode (RGB or RGBA)
            if img_pil.mode == final_image_mode:
                 converted_image_pil = img_pil
            else:
                converted_image_pil = img_pil.convert(final_image_mode)

            image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,] # Add batch dimension

            # Handle mask
            mask_tensor = None
            # Check if original PIL image (before final_image_mode conversion) had alpha
            if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
                if final_image_mode == 'RGBA': # If we are exporting RGBA, use its alpha
                    mask_pil_channel = converted_image_pil.split()[-1]
                else: # If exporting RGB, but original had alpha, still extract alpha from an RGBA version
                    img_rgba_for_mask = img_pil.convert('RGBA')
                    mask_pil_channel = img_rgba_for_mask.split()[-1]
                
                mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,] # Add batch dimension
            else: # No alpha, create a full white mask
                mask_shape = (image_tensor.shape[1], image_tensor.shape[2]) # H, W
                mask_np = np.ones(mask_shape, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_np)[None,] # Add batch dimension

            return image_tensor, mask_tensor, converted_image_pil # Return the PIL image used for tensor (RGB or RGBA)

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found during validation/load: {image_path}")
        except ValueError as ve: # Catch specific validation errors
            raise ve
        except Exception as e:
            raise RuntimeError(f"Error loading/processing image {image_path}: {str(e)}")


    def load_image_or_file(self, folder_path: str, search_subfolders: bool, n_images: int, seed: int, sort: bool, export_with_alpha: bool):
        selected_paths = []
        resolved_path_input = folder_path # Keep original for messages
        
        # Attempt to resolve path using folder_paths if available
        current_check_path = folder_path
        if folder_paths:
            # Check if it's an annotated path first
            annotated_path = folder_paths.get_annotated_filepath(folder_path)
            if annotated_path and os.path.exists(annotated_path):
                current_check_path = annotated_path
            elif not os.path.isabs(folder_path): # If not absolute and not annotated, try input dir
                input_dir = folder_paths.get_input_directory()
                if input_dir:
                    path_in_input_dir = os.path.join(input_dir, folder_path)
                    if os.path.exists(path_in_input_dir):
                        current_check_path = path_in_input_dir
        
        # If still not found, check original path (could be absolute or relative to CWD if folder_paths is not used)
        if not os.path.exists(current_check_path):
             # Final check against the raw input in case it was absolute and folder_paths failed
             if os.path.exists(folder_path):
                 current_check_path = folder_path
             else:
                raise FileNotFoundError(f"Input path '{resolved_path_input}' could not be resolved or found. Checked: '{current_check_path}'")

        if os.path.isfile(current_check_path):
            img_extensions_lower = [ext.lower() for ext in self.IMG_EXTENSIONS]
            if any(current_check_path.lower().endswith(ext) for ext in img_extensions_lower):
                selected_paths = [current_check_path]
            else:
                raise ValueError(f"Input file '{current_check_path}' is not a recognized image type: {self.IMG_EXTENSIONS}")

        elif os.path.isdir(current_check_path):
             try:
                 image_paths_found = self.find_image_files(current_check_path, search_subfolders)
             except FileNotFoundError: # Should be caught by os.path.isdir, but as a safeguard
                 raise FileNotFoundError(f"Directory '{current_check_path}' seems to have disappeared or is not accessible.")
             except Exception as e:
                 raise Exception(f"Error finding image files in '{current_check_path}': {e}")

             valid_image_paths = []
             for f_path in image_paths_found:
                 try:
                     # Basic check with imghdr first (fast)
                     is_image_hdr = imghdr.what(f_path)
                     if is_image_hdr:
                          valid_image_paths.append(f_path)
                          continue
                     else: # If imghdr fails, try PIL verify (slower but more comprehensive for some formats)
                         try:
                             with Image.open(f_path) as img_test:
                                 img_test.verify() # Will raise exception on many invalid image types/corruptions
                             valid_image_paths.append(f_path)
                         except Exception: # PIL verify failed
                              print(f"sokes_nodes.py: Skipping file (PIL verify failed): {f_path}")
                 except Exception as e_outer: # Error with imghdr or other OS issues
                      print(f"sokes_nodes.py: Skipping potentially corrupt/unreadable file: {f_path} - {str(e_outer)}")
             
             if not valid_image_paths:
                 search_scope = "and its subfolders" if search_subfolders else "(subfolders not searched)"
                 raise ValueError(f"No valid images found in folder: {current_check_path} {search_scope}")

             num_available = len(valid_image_paths)
             actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
             
             if actual_n_images == 0 and num_available > 0 :
                 actual_n_images = num_available
             elif actual_n_images == 0 and num_available == 0:
                 raise ValueError(f"No valid images to load from folder: {current_check_path}")

             if actual_n_images < n_images and n_images > 0 :
                  print(f"sokes_nodes.py: Warning: Requested {n_images} images, but only {num_available} were found/valid in '{current_check_path}'. Loading {actual_n_images}.")

             if not sort:
                 random.seed(seed)
                 random.shuffle(valid_image_paths)
                 selected_paths = valid_image_paths[:actual_n_images]
             else:
                 def natural_sort_key(s):
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', os.path.basename(s))]
                 valid_image_paths_sorted = sorted(valid_image_paths, key=natural_sort_key)

                 start_python_index = 0
                 if seed > 0: # Treat seed 0 as "first image regardless of list length", 1 as first, 2 as second etc.
                    start_python_index = (seed -1) % num_available if num_available > 0 else 0
                 
                 selected_paths = []
                 for i in range(actual_n_images):
                     current_idx = (start_python_index + i) % num_available
                     selected_paths.append(valid_image_paths_sorted[current_idx])
        else:
             raise FileNotFoundError(f"Input path '{resolved_path_input}' (resolved to '{current_check_path}') is not a valid file or directory.")

        if not selected_paths:
            raise ValueError("No images were selected to load. Check path, folder contents, or parameters.")

        output_images_tensor_list = []
        output_masks_tensor_list = []
        loaded_paths_list = []
        pil_images_for_preview = [] # Store PIL images (in final RGB/RGBA format) for preview
        first_image_shape_hwc = None

        # Determine final image mode (RGB or RGBA)
        # If export_with_alpha is True, check if ANY selected image has alpha. If so, convert all to RGBA.
        # Otherwise, all to RGB.
        final_image_mode = "RGB"
        if export_with_alpha:
            for image_path_check in selected_paths:
                try:
                    with Image.open(image_path_check) as img_pil_check:
                         if img_pil_check.mode in ('RGBA', 'LA') or \
                            (img_pil_check.mode == 'P' and 'transparency' in img_pil_check.info):
                            final_image_mode = "RGBA"
                            break # Found one with alpha, all will be RGBA
                except Exception as e:
                    print(f"sokes_nodes.py: Warning: Could not pre-check image {os.path.basename(image_path_check)} for alpha: {e}. Assuming no alpha for this image in pre-check.")
        
        print(f"sokes_nodes.py: Final image processing mode for batch: {final_image_mode}")

        for image_path in selected_paths:
             try:
                 image_tensor, mask_tensor, loaded_pil_image = self.validate_and_load_image(image_path, final_image_mode)

                 current_shape_hwc = image_tensor.shape[1:4] # H, W, C
                 if first_image_shape_hwc is None:
                     first_image_shape_hwc = current_shape_hwc
                 elif current_shape_hwc != first_image_shape_hwc and len(selected_paths) > 1:
                     print(f"sokes_nodes.py: ⚠️ Warning: Image {os.path.basename(image_path)} dimensions/channels ({current_shape_hwc}) "
                           f"differ from first image ({first_image_shape_hwc}). Batch may be inconsistent if not handled by subsequent nodes.")

                 output_images_tensor_list.append(image_tensor)
                 output_masks_tensor_list.append(mask_tensor)
                 loaded_paths_list.append(image_path)
                 pil_images_for_preview.append(loaded_pil_image) # Add the loaded PIL image for preview

             except (ValueError, RuntimeError, FileNotFoundError) as e:
                 print(f"sokes_nodes.py: ❌ Skipping image {os.path.basename(image_path)}: {str(e)}")
                 continue
             except Exception as e_unexp: # Catch any other unexpected error during load of a specific image
                 print(f"sokes_nodes.py: ❌ Unexpected error processing image {os.path.basename(image_path)}: {str(e_unexp)}. Skipping.")
                 continue

        if not output_images_tensor_list:
             raise ValueError("No images were successfully loaded into tensors.")

        final_image_batch = torch.cat(output_images_tensor_list, dim=0)
        final_mask_batch = torch.cat(output_masks_tensor_list, dim=0)

        # --- Preview Generation ---
        previews_out_list = []
        if preview_available and pil_images_for_preview: # Use global preview_available flag
            # Ensure the specific subfolder for these previews exists
            preview_subfolder_name = "sokes_nodes_previews" # Specific to this node pack
            full_preview_output_folder = os.path.join(self.output_dir, preview_subfolder_name)
            if not os.path.exists(full_preview_output_folder):
                try:
                    os.makedirs(full_preview_output_folder, exist_ok=True)
                except Exception as e:
                    print(f"sokes_nodes.py: Error creating preview subfolder {full_preview_output_folder}: {e}. Previews may fail.")
                    # Potentially disable previews here if folder creation is critical and fails

            for i, pil_img in enumerate(pil_images_for_preview):
                try:
                    # Use a hash of the original path for a more unique filename
                    # Basename might not be unique if loading from different subfolders with same image name
                    unique_hash = hashlib.sha1(loaded_paths_list[i].encode('utf-8')).hexdigest()[:10]
                    preview_filename = f"preview_{unique_hash}_{i}.png" # Always save as PNG for previews

                    filepath = os.path.join(full_preview_output_folder, preview_filename)
                    pil_img.save(filepath, compress_level=4) # PIL uses compress_level for PNG
                    
                    previews_out_list.append({
                        "filename": preview_filename,
                        "subfolder": preview_subfolder_name, # Use the specific subfolder
                        "type": self.type # Should be "temp"
                    })
                except Exception as e:
                    print(f"sokes_nodes.py: Error generating preview for {os.path.basename(loaded_paths_list[i])}: {e}")
        
        # Return dictionary for ComfyUI
        # The 'ui' key with 'images' is for previews
        # The 'result' tuple matches RETURN_TYPES
        return {"ui": {"images": previews_out_list}, 
                "result": (final_image_batch, final_mask_batch, loaded_paths_list)}


    @classmethod
    def IS_CHANGED(cls, folder_path, search_subfolders, n_images, seed, sort, export_with_alpha):
        # This method is crucial for ComfyUI to know if it needs to re-run the node.
        # It should return a hash or a unique string based on inputs that affect the output.
        
        # Resolve path similarly to how load_image_or_file does for consistency
        current_check_path = folder_path
        if folder_paths:
            annotated_path = folder_paths.get_annotated_filepath(folder_path)
            if annotated_path and os.path.exists(annotated_path):
                current_check_path = annotated_path
            elif not os.path.isabs(folder_path):
                input_dir = folder_paths.get_input_directory()
                if input_dir:
                    path_in_input_dir = os.path.join(input_dir, folder_path)
                    if os.path.exists(path_in_input_dir):
                        current_check_path = path_in_input_dir
        
        if not os.path.exists(current_check_path):
             if os.path.exists(folder_path): # Fallback to original if check path doesn't exist but original does
                 current_check_path = folder_path
             else: # Path truly not found
                 return f"path_not_found_{folder_path}_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"

        file_info_hash_part = ""
        if os.path.isfile(current_check_path):
            try:
                mtime = os.path.getmtime(current_check_path)
                fsize = os.path.getsize(current_check_path)
                file_info_hash_part = f"file_{current_check_path}_{mtime}_{fsize}"
            except Exception as e:
                file_info_hash_part = f"file_error_{current_check_path}_{e}"
        elif os.path.isdir(current_check_path):
            try:
                # For directories, we need to consider the list of files and their mtimes
                # This is a simplified version; a more robust one would walk the dir like find_image_files
                # and hash the relevant file list + mtimes.
                # For now, let's use the directory's own mtime and a quick list of files.
                dir_mtime = os.path.getmtime(current_check_path)
                
                # Simplified: just hash directory mtime and count of top-level items for performance.
                # A full file list hash can be slow for large directories.
                # If deep subfolder changes matter a lot, this might need to be more thorough.
                num_items = 0
                try: # os.listdir can fail on permission issues
                    num_items = len(os.listdir(current_check_path))
                except:
                    pass # keep num_items = 0
                
                file_info_hash_part = f"dir_{current_check_path}_{dir_mtime}_{num_items}"
                if search_subfolders: # If searching subfolders, a more complex hash is needed
                                      # For simplicity, just add a flag for now.
                    # A truly robust subfolder check would walk and hash all relevant file paths/mtimes
                    # This is a placeholder to make it change if search_subfolders changes.
                    # To be more accurate, it should reflect the actual file list that would be found.
                    # For now, this simple approach:
                    # The most accurate way would be to call a light version of find_image_files here
                    # and hash its results (paths and mtimes).
                    # Let's make a more robust version for directories:
                    image_paths_for_hash = []
                    img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS] # Access via cls
                    
                    # Simplified walk for hashing, only considering file names and mtimes
                    # This can still be slow for very large directories.
                    mtimes_list = []
                    try:
                        if search_subfolders:
                            for root, _, files in os.walk(current_check_path):
                                for file_name in files:
                                    if file_name.lower().endswith(tuple(img_extensions_lower)):
                                        try:
                                            full_p = os.path.join(root, file_name)
                                            mtimes_list.append(os.path.getmtime(full_p))
                                        except: pass # Ignore files that disappear or are inaccessible
                        else:
                            if os.path.isdir(current_check_path): # Ensure it's still a dir
                                for f_name in os.listdir(current_check_path):
                                    full_p = os.path.join(current_check_path, f_name)
                                    if os.path.isfile(full_p) and f_name.lower().endswith(tuple(img_extensions_lower)):
                                        try:
                                            mtimes_list.append(os.path.getmtime(full_p))
                                        except: pass
                    except Exception:
                        pass # If listing fails, mtimes_list will be empty

                    hasher = hashlib.sha256()
                    hasher.update(str(sorted(mtimes_list)).encode('utf-8')) # Hash sorted mtimes
                    dir_content_hash = hasher.hexdigest()
                    file_info_hash_part = f"dir_content_{current_check_path}_{dir_content_hash}"

            except Exception as e:
                file_info_hash_part = f"dir_error_{current_check_path}_{e}"
        
        # Combine all relevant input parameters into the hash
        unique_string = f"{file_info_hash_part}_{search_subfolders}_{n_images}_{seed}_{sort}_{export_with_alpha}"
        # Using hashlib for a more compact and consistent hash
        h = hashlib.sha256()
        h.update(unique_string.encode('utf-8'))
        return h.hexdigest()

# END Load Random Image/File with Path and Mask | Sokes 🦬
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
        if not hex_color: return ("Input hex color is empty.",)
        hex_color_proc = hex_color.strip()
        if not hex_color_proc.startswith("#"): hex_color_proc = "#" + hex_color_proc
        if len(hex_color_proc) == 4: # Expand shorthand hex #RGB to #RRGGBB
            hex_color_proc = f"#{hex_color_proc[1]*2}{hex_color_proc[2]*2}{hex_color_proc[3]*2}"
        if len(hex_color_proc) != 7: return (f"Invalid hex format: {hex_color} (processed: {hex_color_proc})",)

        try:
            standard_name = webcolors.hex_to_name(hex_color_proc, spec="css3")
            final_color_name = standard_name
            if not use_css_name:
                 final_color_name = human_readable_map.get(standard_name, standard_name)
            return (final_color_name,)
        except ValueError: # Not an exact CSS3 name
            try:
                requested_rgb = webcolors.hex_to_rgb(hex_color_proc)
                # Ensure RGB values are within 0-255 before creating sRGBColor
                requested_rgb_clamped = tuple(max(0, min(255, c)) for c in requested_rgb)
                requested_lab = convert_color(sRGBColor(*requested_rgb_clamped, is_upscaled=True), LabColor)
            except Exception as e:
                return (f"Invalid input hex '{hex_color_proc}' or conversion error: {e}",)

            min_dist = float('inf')
            closest_name_internal = None # Store the internal name (CSS3 or explicit target key)
            
            # Explicit targets first
            for target_name, target_hex in explicit_targets_for_comparison.items():
                proc_target_hex = target_hex.strip()
                if not proc_target_hex.startswith("#"): proc_target_hex = "#" + proc_target_hex
                if len(proc_target_hex) == 4:
                    proc_target_hex = f"#{proc_target_hex[1]*2}{proc_target_hex[2]*2}{proc_target_hex[3]*2}"
                if len(proc_target_hex) != 7: continue

                try:
                    target_rgb = webcolors.hex_to_rgb(proc_target_hex)
                    target_rgb_clamped = tuple(max(0, min(255, c)) for c in target_rgb)
                    target_lab = convert_color(sRGBColor(*target_rgb_clamped, is_upscaled=True), LabColor)
                    d = delta_e_cie2000(requested_lab, target_lab)
                    if d < min_dist:
                        min_dist = d
                        closest_name_internal = target_name # Use the key from explicit_targets
                except Exception as e:
                    print(f"sokes_nodes.py hex_to_color: Warn: Cannot process explicit target {target_name} ({target_hex}): {e}")

            # Then standard CSS3 colors
            for name, candidate_hex in css3_names_to_hex.items():
                proc_candidate_hex = candidate_hex.strip()
                if not proc_candidate_hex.startswith("#"): proc_candidate_hex = "#" + proc_candidate_hex
                if len(proc_candidate_hex) == 4:
                     proc_candidate_hex = f"#{proc_candidate_hex[1]*2}{proc_candidate_hex[2]*2}{proc_candidate_hex[3]*2}"
                if len(proc_candidate_hex) != 7: continue

                try:
                    cand_rgb = webcolors.hex_to_rgb(proc_candidate_hex)
                    cand_rgb_clamped = tuple(max(0, min(255, c)) for c in cand_rgb)
                    cand_lab = convert_color(sRGBColor(*cand_rgb_clamped, is_upscaled=True), LabColor)
                    d = delta_e_cie2000(requested_lab, cand_lab)
                    if d < min_dist:
                        min_dist = d
                        closest_name_internal = name # This is a CSS3 name
                except Exception as e:
                    print(f"sokes_nodes.py hex_to_color: Warn: Error processing CSS3 candidate {name} ({candidate_hex}): {e}")
            
            if closest_name_internal is None: return ("Could not find any closest color match.",)

            # Map the found internal name (CSS3 or explicit key) to human-readable if not use_css_name
            if use_css_name:
                # If closest was from explicit_targets, and we need CSS name, we might need to find its CSS name if it has one
                # or just return the explicit target's name if it's deemed "CSS-like enough"
                # For simplicity now, if use_css_name is true, and closest_name_internal is from explicit_targets,
                # we will return closest_name_internal. If it was a CSS3 name, that's fine too.
                final_color_name = closest_name_internal
            else: # Map to human readable
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
        # Ensure min <= max
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
        # IS_CHANGED should reflect anything that would change the output if the node were re-run
        # For a random number generator, the seed is paramount. Min/max also change the range.
        h = hashlib.sha256()
        h.update(str(minimum).encode('utf-8'))
        h.update(str(maximum).encode('utf-8'))
        h.update(str(seed).encode('utf-8'))
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

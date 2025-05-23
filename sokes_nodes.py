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
# import imghdr # load random image # Already imported above

# Might need to: pip install webcolors colormath
import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

from .sokes_color_maps import css3_names_to_hex, css3_hex_to_names, human_readable_map, explicit_targets_for_comparison

# import numpy as np # Already imported above
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
# START Load Random Image with Path and Mask | Sokes 🦬

# Try importing folder_paths and preview_images if available (for ComfyUI integration)
try:
    import folder_paths
except ImportError:
    print("sokes_nodes.py: 'folder_paths' module not found. Path resolution might be limited.")
    folder_paths = None # Set to None if not available

try:
    # Import the preview handling module from ComfyUI utilities
    from server import PromptServer
    from comfy.utils import FONT_PATH # To check if we are in ComfyUI
    import nodes # To get the preview_widget decorator potentially
    preview_available = True
    # Basic check if we're likely running within ComfyUI server context
    try:
        PromptServer.instance
    except:
        preview_available = False
        print("sokes_nodes.py: PromptServer instance not found. Output preview disabled.")

except ImportError:
    preview_available = False
    print("sokes_nodes.py: Could not import 'server' or 'comfy.utils'. Output preview disabled.")

class load_random_image_sokes:
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory() if folder_paths else "temp_sokes"
        self.type = "temp"
        if not os.path.exists(self.output_dir) and self.type == "temp":
             try: os.makedirs(self.output_dir)
             except: print(f"Warning: Could not create temp directory {self.output_dir}")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ".", "multiline": False}),
                "search_subfolders": ("BOOLEAN", {"default": False}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # If sort=False, used for random shuffle. If sort=True, 0 or 1 selects 1st image, 2 selects 2nd, etc. (1-based index).
                "sort": ("BOOLEAN", {"default": False}),
                "export_with_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image_or_file"

    OUTPUT_NODE = True

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
                all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
                image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(tuple(img_extensions_lower))]
        except FileNotFoundError:
             raise FileNotFoundError(f"Directory not found: {folder_path}")
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder_path}': {e}")
        return image_paths

    def validate_and_load_image(self, image_path, final_image_mode):
        try:
            img_pil = Image.open(image_path)
            img_pil = ImageOps.exif_transpose(img_pil)

            is_image_hdr = imghdr.what(image_path)
            if not is_image_hdr:
                try:
                    img_pil.verify()
                except Exception as pil_e:
                    raise ValueError(f"Invalid or corrupt image file (PIL verify failed): {image_path} - {str(pil_e)}")

            if img_pil.mode == final_image_mode:
                 converted_image_pil = img_pil
            else:
                converted_image_pil = img_pil.convert(final_image_mode)

            image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            mask_tensor = None
            if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
                if img_pil.mode != 'RGBA':
                    img_rgba_for_mask = img_pil.convert('RGBA')
                else:
                    img_rgba_for_mask = img_pil
                mask_pil_channel = img_rgba_for_mask.split()[-1]
                mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)[None,]
            else:
                mask_shape = (image_tensor.shape[1], image_tensor.shape[2])
                mask_np = np.ones(mask_shape, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_np)[None,]

            return image_tensor, mask_tensor, img_pil

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading/processing image {image_path}: {str(e)}")


    def load_image_or_file(self, folder_path, search_subfolders, n_images, seed, sort, export_with_alpha):
        selected_paths = []
        # input_is_file = False # Not strictly needed with current logic flow
        resolved_path = folder_path

        is_file_check_path = resolved_path
        if folder_paths and not os.path.isabs(resolved_path) and (not os.path.exists(resolved_path)):
             maybe_resolved = folder_paths.get_annotated_filepath(resolved_path)
             if maybe_resolved and os.path.exists(maybe_resolved):
                 is_file_check_path = maybe_resolved
             else:
                input_dir = folder_paths.get_input_directory()
                if input_dir:
                    maybe_resolved_input = os.path.join(input_dir, resolved_path)
                    if os.path.exists(maybe_resolved_input):
                        is_file_check_path = maybe_resolved_input

        if os.path.isfile(is_file_check_path):
            img_extensions_lower = [ext.lower() for ext in self.IMG_EXTENSIONS]
            if any(is_file_check_path.lower().endswith(ext) for ext in img_extensions_lower):
                selected_paths = [is_file_check_path]
                # input_is_file = True
                resolved_path = is_file_check_path
                #print(f"Input is a single file: {resolved_path}")
            else:
                raise ValueError(f"Input file '{resolved_path}' is not a recognized image type: {self.IMG_EXTENSIONS}")

        elif os.path.isdir(is_file_check_path):
             resolved_path = is_file_check_path
             #print(f"Input is a folder: {resolved_path}. Searching (subfolders: {search_subfolders})...")
             try:
                 image_paths_found = self.find_image_files(resolved_path, search_subfolders)
             except Exception as e:
                 raise Exception(f"Error finding image files in '{resolved_path}': {e}")

             valid_image_paths = []
             for f_path in image_paths_found:
                 try:
                     is_image_hdr = imghdr.what(f_path)
                     if is_image_hdr:
                          valid_image_paths.append(f_path)
                          continue
                     else:
                         try:
                             with Image.open(f_path) as img_test:
                                 img_test.verify()
                             valid_image_paths.append(f_path)
                         except Exception as pil_e:
                              print(f"Skipping file in folder (PIL verify failed): {f_path} - {str(pil_e)}")
                 except Exception as e:
                      print(f"Skipping potentially corrupt/unreadable file in folder: {f_path} - {str(e)}")

             if not valid_image_paths:
                 search_scope = "and its subfolders" if search_subfolders else "(subfolders not searched)"
                 raise ValueError(f"No valid images found in folder: {resolved_path} {search_scope}")

             num_available = len(valid_image_paths)
             actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
             if actual_n_images == 0 and num_available > 0 : # if n_images was 0, actual_n_images becomes num_available
                 actual_n_images = num_available
             elif actual_n_images == 0 and num_available == 0:
                 raise ValueError(f"No valid images to load from folder: {resolved_path}")


             if actual_n_images < n_images and n_images > 0 :
                  print(f"Warning: Requested {n_images} images, but only {num_available} were found/valid in '{resolved_path}'. Loading {actual_n_images}.")

             if not sort:
                 random.seed(seed)
                 random.shuffle(valid_image_paths)
                 selected_paths = valid_image_paths[:actual_n_images]
                 #if selected_paths:
                     #print(f"Sort=False: Selected {len(selected_paths)} image(s) randomly using seed {seed}.")
             else: # sort is True
                 def natural_sort_key(s):
                    # Sort by basename for natural file order
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', os.path.basename(s))]
                 
                 valid_image_paths = sorted(valid_image_paths, key=natural_sort_key)

                 # Interpret seed as 1-based index for sorted list
                 # seed = 0 or 1 maps to the first image (index 0)
                 # seed = 2 maps to the second image (index 1), etc.
                 if seed <= 1: # UI seed 0 or 1
                     start_python_index = 0
                 else: # UI seed > 1
                     start_python_index = (seed - 1) % num_available
                
                 selected_paths = []
                 for i in range(actual_n_images):
                     current_index = (start_python_index + i) % num_available
                     selected_paths.append(valid_image_paths[current_index])
                 
                 #if selected_paths:
                 #   if n_images == 1:
                 #       #print(f"Sort=True, n_images=1: Selecting sorted image at 1-based index {start_python_index + 1} (seed {seed}) from {num_available} images: {os.path.basename(selected_paths[0])}")
                 #   else:
                 #       #print(f"Sort=True, n_images={len(selected_paths)}: Selecting batch starting at 1-based index {start_python_index + 1} (seed {seed}), with wrapping.")
        else:
             raise FileNotFoundError(f"Input path '{folder_path}' (resolved check path: '{is_file_check_path}') is not a valid file or directory.")

        if not selected_paths: # Should be caught earlier, but as a final check
            raise ValueError("No images were selected to load. Check path, folder contents, or parameters.")

        output_images = []
        output_masks = []
        loaded_paths = []
        preview_pil_images = []
        first_image_shape_hw = None

        force_batch_to_rgba = False
        if export_with_alpha and selected_paths:
            for image_path_check in selected_paths:
                try:
                    with Image.open(image_path_check) as img_pil_check:
                         if img_pil_check.mode in ('RGBA', 'LA') or (img_pil_check.mode == 'P' and 'transparency' in img_pil_check.info):
                            force_batch_to_rgba = True
                            break
                except Exception as e:
                    print(f"⚠️ Warning: Could not pre-check image {image_path_check} for alpha: {e}. Assuming no alpha.")

        final_image_mode = "RGB"
        if export_with_alpha and force_batch_to_rgba:
            final_image_mode = "RGBA"
        elif not export_with_alpha: # or (export_with_alpha and not force_batch_to_rgba)
            final_image_mode = "RGB"

        for image_path in selected_paths:
             try:
                 image_tensor, mask_tensor, loaded_pil_image = self.validate_and_load_image(image_path, final_image_mode)

                 current_shape_hw = image_tensor.shape[1:3]
                 if first_image_shape_hw is None:
                     first_image_shape_hw = current_shape_hw
                 elif current_shape_hw != first_image_shape_hw and len(selected_paths) > 1:
                     print(f"⚠️ Warning: Image {os.path.basename(image_path)} dimensions ({current_shape_hw}) "
                           f"differ from first image ({first_image_shape_hw}). Batch may be inconsistent.")

                 output_images.append(image_tensor)
                 output_masks.append(mask_tensor)
                 loaded_paths.append(image_path)
                 preview_pil_images.append(loaded_pil_image)

             except (ValueError, RuntimeError, FileNotFoundError) as e:
                 print(f"❌ Skipping image {image_path}: {str(e)}")
                 continue
             except Exception as e:
                 print(f"❌ Unexpected error processing image {image_path}: {str(e)}. Skipping.")
                 continue

        if not output_images:
             raise ValueError("No images were successfully loaded.")

        final_image_batch = torch.cat(output_images, dim=0)
        final_mask_batch = torch.cat(output_masks, dim=0)

        previews_out = []
        if preview_available and preview_pil_images:
            for i, pil_img in enumerate(preview_pil_images):
                try:
                    basename = os.path.basename(loaded_paths[i])
                    subfolder = "sokes_previews"
                    filename = f"preview_{hashlib.sha1(loaded_paths[i].encode()).hexdigest()}_{i}.png"

                    full_output_folder = os.path.join(self.output_dir, subfolder)
                    if not os.path.exists(full_output_folder):
                         os.makedirs(full_output_folder, exist_ok=True)

                    filepath = os.path.join(full_output_folder, filename)
                    pil_img.save(filepath, compress_level=4)
                    previews_out.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": self.type
                    })
                except Exception as e:
                    print(f"Error generating preview for {loaded_paths[i]}: {e}")

        result = {
            "ui": {"images": previews_out},
            "result": (final_image_batch, final_mask_batch, loaded_paths)
        }
        return result


    @classmethod
    def IS_CHANGED(cls, folder_path, search_subfolders, n_images, seed, sort, export_with_alpha):
        resolved_path = folder_path

        if folder_paths and not os.path.isabs(resolved_path) and (not os.path.exists(resolved_path)):
             maybe_resolved = folder_paths.get_annotated_filepath(resolved_path)
             if maybe_resolved and os.path.exists(maybe_resolved):
                 resolved_path = maybe_resolved
             else:
                 input_dir = folder_paths.get_input_directory()
                 if input_dir:
                     maybe_resolved_input = os.path.join(input_dir, resolved_path)
                     if os.path.exists(maybe_resolved_input):
                         resolved_path = maybe_resolved_input
        try:
            if os.path.isfile(resolved_path):
                img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS]
                if any(resolved_path.lower().endswith(ext) for ext in img_extensions_lower):
                    # is_file = True
                    mtime = os.path.getmtime(resolved_path)
                    # For single file, n_images, seed, sort, search_subfolders are ignored by current loading logic
                    return f"file_{resolved_path}_{mtime}_{export_with_alpha}"
                else:
                     return f"invalid_file_type_{resolved_path}"

            elif os.path.isdir(resolved_path):
                # is_dir = True
                image_paths = []
                img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS]
                if search_subfolders:
                     for root, _, files in os.walk(resolved_path):
                        for file_name in files: # Renamed 'file' to 'file_name' to avoid conflict
                             full_path = os.path.join(root, file_name)
                             if os.path.isfile(full_path) and file_name.lower().endswith(tuple(img_extensions_lower)):
                                image_paths.append(full_path)
                else:
                    all_files = [os.path.join(resolved_path, f) for f in os.listdir(resolved_path)]
                    image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(tuple(img_extensions_lower))]

                if not image_paths:
                    # seed is relevant as it's part of the input state
                    return f"no_images_in_dir_{resolved_path}_{search_subfolders}_{n_images}_{sort}_{export_with_alpha}_{seed}"

                # Sort lexicographically for consistent file list hash
                image_paths_for_hash = sorted(image_paths)
                mtimes = [os.path.getmtime(f) for f in image_paths_for_hash]
                file_info_tuple = tuple(zip(image_paths_for_hash, mtimes))
                hasher = hashlib.sha256()
                hasher.update(str(file_info_tuple).encode('utf-8'))
                file_list_hash = hasher.hexdigest()
                
                # Seed is now relevant for both sort=True (start index) and sort=False (shuffle)
                return f"dir_{file_list_hash}_{search_subfolders}_{n_images}_{sort}_{export_with_alpha}_{seed}"
            else:
                 return f"path_not_found_{resolved_path}"
        except Exception as e:
             print(f"sokes_nodes.py IS_CHANGED Error: {e}")
             return float("NaN")

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
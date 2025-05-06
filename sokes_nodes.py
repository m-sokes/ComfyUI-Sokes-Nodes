from datetime import datetime # datetime
import re # regex
import os # for load random image
import random # for load random image
import imghdr # to check image by header vs extension
import torch # to resize images
import numpy as np # for image manipulation in torch
import cv2 # for image processing
from PIL import Image, ImageOps # load random image, orient image

# Try to import folder_paths, ComfyUI should make this available
try:
    import folder_paths
except ImportError:
    # Create a dummy folder_paths if not found (e.g. running outside ComfyUI)
    # This allows the script to be parsed, but image loading might fail
    # if paths are relative and not resolvable without ComfyUI's context.
    print("sokes_nodes.py: folder_paths module not found. Image loading might be limited to absolute paths.")
    class DummyFolderPaths:
        def get_input_directory(self):
            # Try to guess a common input directory, or return None
            if os.path.isdir("input"): return "input"
            if os.path.isdir("../input"): return "../input" # If nodes are in subfolder
            return None

        def get_annotated_filepath(self, file_path):
            # In a dummy scenario, just return the file_path if it's part of a common structure
            # or assume it's absolute/resolvable as is.
            input_dir = self.get_input_directory()
            if input_dir and os.path.exists(os.path.join(input_dir, file_path)):
                return os.path.join(input_dir, file_path)
            return file_path # Fallback

        def get_folder_paths(self, folder_name): #Placeholder for other potential uses
            if folder_name == "input" and self.get_input_directory():
                return [self.get_input_directory()]
            return []

    folder_paths = DummyFolderPaths()


# Current Date | Sokes 🦬

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
        formatted = re.sub(r'(?i)([YMD])\1*', lambda m: m.group().upper(), date_format)


        # Replace in order: longest first to prevent partial replacements
        # Year
        formatted = formatted.replace("YYYY", now.strftime("%Y"))
        formatted = formatted.replace("YY", now.strftime("%y"))
        
        # Month 
        formatted = formatted.replace("MM", now.strftime("%m"))  # Zero-padded
        formatted = formatted.replace("M", str(now.month)) # No zero-pad (strftime %-m is platform dependent)
        
        # Day
        formatted = formatted.replace("DD", now.strftime("%d"))  # Zero-padded
        formatted = formatted.replace("D", str(now.day))    # No zero-pad (strftime %-d is platform dependent)

        return (formatted,)

    @classmethod
    def IS_CHANGED(cls, date_format):
        # Force re-execution
        return (datetime.now().timestamp(),)


##############################################################
# START: Latent Input Swtich x9 | Sokes 🦬

class latent_input_switch_9x_sokes:
    @classmethod
    def INPUT_TYPES(cls): # Changed s to cls
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
    # OUTPUT_NODE = True # Typically not needed for a switch, can be removed unless specific reason
    CATEGORY = "Sokes 🦬"

    def latent_input_switch_9x_sokes(self, latent_select, latent_0, latent_1=None, latent_2=None, latent_3=None, latent_4=None, latent_5=None, latent_6=None, latent_7=None, latent_8=None):
        latents = [latent_0, latent_1, latent_2, latent_3, latent_4, latent_5, latent_6, latent_7, latent_8]
        selected_index = int(round(latent_select))

        if 0 <= selected_index < len(latents) and latents[selected_index] is not None:
            return (latents[selected_index],)
        else:
            # Fallback to latent_0 if selection is out of bounds or selected latent is None
            if latent_0 is not None:
                return (latent_0,)
            else:
                # This case should ideally be handled by ComfyUI's graph validation
                # or you might want to raise an error or return a dummy latent if possible.
                # For now, returning None might cause issues downstream.
                # A better fallback might be needed depending on ComfyUI's handling of None latents.
                # However, latent_0 is required, so it should not be None.
                # If the selected optional latent is None, then falling back to latent_0 is the current logic.
                # If latent_0 itself IS None (which shouldn't happen due to "required"),
                # then this is an issue.
                # The original logic returns latent_0 which works fine if latent_0 is guaranteed.
                return (latent_0,)


# END: Latent Input Swtich x9 | Sokes 🦬
##############################################################

##############################################################
# START: Replace Text with Regex | Sokes 🦬

class replace_text_regex_sokes:
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}), # Comfy preferred: "default": ""
            "regex_pattern": ("STRING", {"multiline": False, "default": ""}),
            "new": ("STRING", {"multiline": False, "default": ""})
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fn_replace_text_regex_sokes"
    CATEGORY = "Sokes 🦬"

    # @ staticmethod # Static method is fine
    def fn_replace_text_regex_sokes(self, regex_pattern, new, text): # self if not static
        return (re.sub(regex_pattern, new, text),)


# END: Replace Text with Regex | Sokes 🦬
##############################################################

##############################################################
# START: Load Random Image with Path and Mask | Sokes 🦬
 
class load_random_image_sokes:
    # IMG_EXTENSIONS as a class variable
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        pass # No need for self.img_extensions if using class variable
 
    @classmethod
    def INPUT_TYPES(cls): # Changed s to cls
        return {
            "required": {
                "folder": ("STRING", {"default": "."}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sort": ("BOOLEAN", {"default": False}),
            }
        }
 
    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image"
 
    def load_image(self, folder, n_images, seed, sort):
        resolved_folder_path = folder
        if not os.path.isdir(folder):
            try:
                # Use imported folder_paths
                input_dir = folder_paths.get_input_directory()
                annotated_path = folder_paths.get_annotated_filepath(folder)

                if annotated_path and os.path.isdir(annotated_path):
                    resolved_folder_path = annotated_path
                elif input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                    resolved_folder_path = os.path.join(input_dir, folder)
                else:
                    # If still not a directory, will be caught by the check below
                    pass
            except NameError:
                print(f"sokes_nodes.py: 'folder_paths' module not available for resolving '{folder}'. Using path as is.")
            except Exception as e:
                print(f"sokes_nodes.py: Error trying to resolve folder '{folder}' using folder_paths: {e}. Using path as is.")
        
        if not os.path.isdir(resolved_folder_path):
            raise FileNotFoundError(f"Folder '{resolved_folder_path}' (resolved from '{folder}') not found or not a directory.")
        
        folder = resolved_folder_path # Use the resolved path from now on

        try:
            all_files = [os.path.join(folder, f) for f in os.listdir(folder)]
            image_paths = [f for f in all_files if os.path.isfile(f)]
            # Use class variable IMG_EXTENSIONS
            image_paths = [f for f in image_paths if any([f.lower().endswith(ext) for ext in self.IMG_EXTENSIONS])]
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder}': {e}")
 
        valid_image_paths = []
        for f_path in image_paths:
            try:
                is_image = imghdr.what(f_path)
                if not is_image:
                    try:
                        img_test = Image.open(f_path)
                        img_test.verify()
                        is_image = True
                    except Exception:
                        is_image = False
                if is_image:
                    valid_image_paths.append(f_path)
            except Exception as e:
                 print(f"Skipping potentially corrupt image: {f_path} - {str(e)}")
 
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")
 
        if not sort:
            random.seed(seed)
            random.shuffle(valid_image_paths)
        else:
             valid_image_paths = sorted(valid_image_paths)
 
        num_available = len(valid_image_paths)
        actual_n_images = min(n_images, num_available)
        if actual_n_images < n_images and n_images > 0:
             print(f"Warning: Requested {n_images} images, but only {num_available} were found/valid in '{folder}'. Loading {actual_n_images}.")
        elif n_images <= 0: # Treat 0 or negative as "load all available"
             actual_n_images = num_available
 
        selected_paths = valid_image_paths[:actual_n_images]
 
        output_images = []
        output_masks = []
        loaded_paths = []
        first_image_shape = None
 
        for image_path in selected_paths:
            try:
                img_pil = Image.open(image_path)
                img_pil = ImageOps.exif_transpose(img_pil)
 
                image_rgb_pil = img_pil.convert("RGB")
                image_np = np.array(image_rgb_pil).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
 
                mask_tensor = None
                if 'A' in img_pil.getbands() or 'a' in img_pil.getbands():
                    mask_pil_channel = img_pil.getchannel('A')
                    mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np)[None,]
                else:
                    mask_shape = (image_tensor.shape[1], image_tensor.shape[2])
                    mask_np = np.ones(mask_shape, dtype=np.float32)
                    mask_tensor = torch.from_numpy(mask_np)[None,]
 
                current_shape = image_tensor.shape[1:3]
                if first_image_shape is None:
                    first_image_shape = current_shape
                elif current_shape != first_image_shape:
                    print(f"⚠️ Warning: Image {os.path.basename(image_path)} has different dimensions ({current_shape}) than the first image ({first_image_shape}). Batching may fail or produce unexpected results downstream. Consider resizing images beforehand.")
 
                output_images.append(image_tensor)
                output_masks.append(mask_tensor)
                loaded_paths.append(image_path)
 
            except Exception as e:
                print(f"❌ Error loading or processing image {image_path}: {str(e)}. Skipping.")
                continue
 
        if not output_images:
            raise ValueError("No images were successfully loaded after processing.")
 
        final_image_batch = torch.cat(output_images, dim=0)
        final_mask_batch = torch.cat(output_masks, dim=0)
 
        return (final_image_batch, final_mask_batch, loaded_paths)
 
    @classmethod
    def IS_CHANGED(cls, folder, n_images, seed, sort): # Changed s to cls
        resolved_folder_path = folder
        try:
            # Use imported folder_paths
            if not os.path.isdir(folder):
                annotated_path = folder_paths.get_annotated_filepath(folder)
                if annotated_path and os.path.isdir(annotated_path):
                    resolved_folder_path = annotated_path
                else:
                    input_dir = folder_paths.get_input_directory()
                    if input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                        resolved_folder_path = os.path.join(input_dir, folder)
        except NameError:
            # folder_paths not defined, use path as is for hashing.
            # IS_CHANGED might be less effective but won't crash.
            pass
        except Exception as e:
            print(f"sokes_nodes.py IS_CHANGED: Error during folder_path resolution: {e}")
            # Fallback to original folder if resolution fails, for hashing purposes
            pass

        if not os.path.isdir(resolved_folder_path):
            return float("NaN") # Indicate path not found, should re-evaluate

        try:
            all_files = [os.path.join(resolved_folder_path, f) for f in os.listdir(resolved_folder_path)]
            image_paths = [f for f in all_files if os.path.isfile(f)]
            # Use class variable IMG_EXTENSIONS
            img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS]
            image_paths = [f for f in image_paths if f.lower().endswith(tuple(img_extensions_lower))]
            
            if not image_paths: # No images found
                return f"no_images_in_{resolved_folder_path}_{seed}_{n_images}_{sort}"

            image_paths = sorted(image_paths) # Sort for consistent hash base
            mtimes = [os.path.getmtime(f) for f in image_paths]
            file_info_tuple = tuple(zip(image_paths, mtimes))
            file_list_hash = hash(file_info_tuple)
 
            # The 'sort' parameter affects image selection order, 'seed' affects it if not sorted
            if not sort:
                return f"{file_list_hash}_{seed}_{n_images}"
            else: # If sorted, seed doesn't influence order of already sorted files
                return f"{file_list_hash}_{n_images}"
        except Exception as e:
             print(f"sokes_nodes.py IS_CHANGED Error: {e}")
             return float("NaN") # Fallback to always re-execute on error
  
# END: Load Random Image with Path and Mask | Sokes 🦬
##############################################################


# These functions are not currently used by any node in this file.
# They can be kept for future use or removed if not needed.
def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))


def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]
    if w/h > aspect_ratio:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        crop = img[:, left:right]
    else:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2
        crop = img[top:bottom, :]
    return crop


def create_same_sized_crops(imgs, target_n_pixels=2048**2):
    assert len(imgs) > 1
    imgs_np = [np.array(img) for img in imgs] # Ensure they are numpy arrays
    
    aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs_np]
    final_aspect_ratio = np.mean(aspect_ratios)
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs_np]

    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = final_h * final_aspect_ratio
    final_h = round_to_nearest_multiple(final_h, 8)
    final_w = round_to_nearest_multiple(final_w, 8)

    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    # To return PIL Images:
    # resized_imgs_pil = [Image.fromarray(img) for img in resized_imgs] # Assuming img is uint8 after resize
    return resized_imgs

##############################################################
# Node Mappings

NODE_CLASS_MAPPINGS = {
    "Latent Switch x9 | sokes 🦬": latent_input_switch_9x_sokes,
    "Current Date | sokes 🦬": current_date_sokes,
    "Replace Text with RegEx | sokes 🦬": replace_text_regex_sokes,
    "Load Random Image | sokes 🦬": load_random_image_sokes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Current Date | sokes 🦬": "Current Date 🦬",
    "Latent Switch x9 | sokes 🦬": "Latent Switch x9 🦬",
    "Replace Text with RegEx | sokes 🦬": "Replace Text with RegEx 🦬",
    "Load Random Image | sokes 🦬": "Load Random Image 🦬",
}

# A manifest for ComfyUI
MANIFEST = {
    "name": "Sokes Nodes",
    "version": (1,0,5),
    "author": "Sokes",
    "project": "https://github.com/m-sokes/ComfyUI-Sokes-Nodes/upload/main",
    "description": "A collection of utility nodes for ComfyUI by Sokes.",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'MANIFEST']
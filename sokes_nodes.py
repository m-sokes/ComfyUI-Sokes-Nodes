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
    print("sokes_nodes.py: folder_paths module not found. Image loading might be limited to absolute paths.")
    class DummyFolderPaths:
        def get_input_directory(self):
            if os.path.isdir("input"): return "input"
            if os.path.isdir("../input"): return "../input"
            return None
        def get_annotated_filepath(self, file_path):
            input_dir = self.get_input_directory()
            if input_dir and os.path.exists(os.path.join(input_dir, file_path)):
                return os.path.join(input_dir, file_path)
            return file_path
        def get_folder_paths(self, folder_name):
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
        now = datetime.now()
        formatted = re.sub(r'(?i)([YMD])\1*', lambda m: m.group().upper(), date_format)
        formatted = formatted.replace("YYYY", now.strftime("%Y"))
        formatted = formatted.replace("YY", now.strftime("%y"))
        formatted = formatted.replace("MM", now.strftime("%m"))
        formatted = formatted.replace("M", str(now.month))
        formatted = formatted.replace("DD", now.strftime("%d"))
        formatted = formatted.replace("D", str(now.day))
        return (formatted,)
    @classmethod
    def IS_CHANGED(cls, date_format):
        return (datetime.now().timestamp(),)

# Latent Input Swtich x9 | Sokes 🦬
class latent_input_switch_9x_sokes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_select": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "latent_0": ("LATENT",),
            },
            "optional": { f"latent_{i}": ("LATENT",) for i in range(1, 9) },
        }
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "latent_input_switch_9x_sokes"
    CATEGORY = "Sokes 🦬"
    def latent_input_switch_9x_sokes(self, latent_select, **kwargs):
        # kwargs will contain latent_0, latent_1, ..., latent_8
        # Ensure latent_0 is always present as it's required
        latents = [kwargs.get(f"latent_{i}") for i in range(9)] # latent_0 to latent_8
        if 'latent_0' in kwargs: # Prioritize the named required latent_0
            latents[0] = kwargs['latent_0']

        selected_index = int(round(latent_select))
        if 0 <= selected_index < len(latents) and latents[selected_index] is not None:
            return (latents[selected_index],)
        elif latents[0] is not None: # Fallback to latent_0
            return (latents[0],)
        else: # Should not happen if latent_0 is required and provided
            raise ValueError("latent_0 is required and was not provided or is None.")

# Replace Text with Regex | Sokes 🦬
class replace_text_regex_sokes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "regex_pattern": ("STRING", {"multiline": False, "default": ""}),
            "new": ("STRING", {"multiline": False, "default": ""})
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "fn_replace_text_regex_sokes"
    CATEGORY = "Sokes 🦬"
    def fn_replace_text_regex_sokes(self, regex_pattern, new, text):
        return (re.sub(regex_pattern, new, text),)

##############################################################
# START: Load Random Image with Path and Mask | Sokes 🦬
 
class load_random_image_sokes:
    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]

    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "."}),
                "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sort": ("BOOLEAN", {"default": False}),
                "export_with_alpha": ("BOOLEAN", {"default": False}), # New Toggle
            }
        }
 
    CATEGORY = "Sokes 🦬/Loaders"
    RETURN_TYPES = ("IMAGE", "MASK", "LIST")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image"
 
    def load_image(self, folder, n_images, seed, sort, export_with_alpha): # Added export_with_alpha
        resolved_folder_path = folder
        if not os.path.isdir(folder):
            try:
                input_dir = folder_paths.get_input_directory()
                annotated_path = folder_paths.get_annotated_filepath(folder)
                if annotated_path and os.path.isdir(annotated_path):
                    resolved_folder_path = annotated_path
                elif input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                    resolved_folder_path = os.path.join(input_dir, folder)
            except NameError:
                print(f"sokes_nodes.py: 'folder_paths' module not available for resolving '{folder}'. Using path as is.")
            except Exception as e:
                print(f"sokes_nodes.py: Error trying to resolve folder '{folder}' using folder_paths: {e}. Using path as is.")
        
        if not os.path.isdir(resolved_folder_path):
            raise FileNotFoundError(f"Folder '{resolved_folder_path}' (resolved from '{folder}') not found or not a directory.")
        
        folder = resolved_folder_path

        try:
            all_files = [os.path.join(folder, f) for f in os.listdir(folder)]
            image_paths_in_folder = [f for f in all_files if os.path.isfile(f)]
            image_paths_in_folder = [f for f in image_paths_in_folder if any([f.lower().endswith(ext) for ext in self.IMG_EXTENSIONS])]
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder}': {e}")
 
        valid_image_paths = []
        for f_path in image_paths_in_folder:
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
        actual_n_images = min(n_images, num_available) if n_images > 0 else num_available
        if actual_n_images < n_images and n_images > 0 :
             print(f"Warning: Requested {n_images} images, but only {num_available} were found/valid in '{folder}'. Loading {actual_n_images}.")
 
        selected_paths = valid_image_paths[:actual_n_images]
 
        output_images = []
        output_masks = []
        loaded_paths = []
        first_image_shape_hw = None # For H, W consistency
        
        # --- Determine batch output mode if export_with_alpha is True ---
        force_batch_to_rgba = False
        if export_with_alpha and selected_paths:
            for image_path_check in selected_paths:
                try:
                    img_pil_check = Image.open(image_path_check)
                    if 'A' in img_pil_check.getbands() or 'a' in img_pil_check.getbands():
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
                img_pil = ImageOps.exif_transpose(img_pil)
 
                # Convert image for IMAGE output based on determined mode
                converted_image_pil = img_pil.convert(final_image_mode)
                image_np = np.array(converted_image_pil).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # Shape will be [1, H, W, 3] or [1, H, W, 4]
 
                # Mask Processing (always from original PIL image's alpha)
                mask_tensor = None
                if 'A' in img_pil.getbands() or 'a' in img_pil.getbands():
                    mask_pil_channel = img_pil.getchannel('A')
                    mask_np = np.array(mask_pil_channel).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np)[None,]
                else:
                    # Use shape from the (potentially 3 or 4 channel) image_tensor for H, W
                    mask_shape = (image_tensor.shape[1], image_tensor.shape[2])
                    mask_np = np.ones(mask_shape, dtype=np.float32)
                    mask_tensor = torch.from_numpy(mask_np)[None,]
 
                # Batch Consistency Check (H, W dimensions)
                current_shape_hw = image_tensor.shape[1:3] # H, W
                if first_image_shape_hw is None:
                    first_image_shape_hw = current_shape_hw
                elif current_shape_hw != first_image_shape_hw:
                    # This warning remains important. Resizing would be needed for true batching if sizes differ.
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
                continue
 
        if not output_images:
            raise ValueError("No images were successfully loaded after processing.")
 
        # torch.cat should now work as channels are consistent across the batch for output_images
        final_image_batch = torch.cat(output_images, dim=0)
        final_mask_batch = torch.cat(output_masks, dim=0)
 
        return (final_image_batch, final_mask_batch, loaded_paths)
 
    @classmethod
    def IS_CHANGED(cls, folder, n_images, seed, sort, export_with_alpha): # Added export_with_alpha
        resolved_folder_path = folder
        try:
            if not os.path.isdir(folder):
                annotated_path = folder_paths.get_annotated_filepath(folder)
                if annotated_path and os.path.isdir(annotated_path):
                    resolved_folder_path = annotated_path
                else:
                    input_dir = folder_paths.get_input_directory()
                    if input_dir and os.path.isdir(os.path.join(input_dir, folder)):
                        resolved_folder_path = os.path.join(input_dir, folder)
        except NameError: pass
        except Exception as e:
            print(f"sokes_nodes.py IS_CHANGED: Error during folder_path resolution: {e}")
            pass

        if not os.path.isdir(resolved_folder_path):
            return float("NaN") 

        try:
            all_files = [os.path.join(resolved_folder_path, f) for f in os.listdir(resolved_folder_path)]
            image_paths = [f for f in all_files if os.path.isfile(f)]
            img_extensions_lower = [ext.lower() for ext in cls.IMG_EXTENSIONS]
            image_paths = [f for f in image_paths if f.lower().endswith(tuple(img_extensions_lower))]
            
            if not image_paths:
                return f"no_images_in_{resolved_folder_path}_{seed}_{n_images}_{sort}_{export_with_alpha}"

            image_paths = sorted(image_paths) 
            mtimes = [os.path.getmtime(f) for f in image_paths]
            file_info_tuple = tuple(zip(image_paths, mtimes))
            file_list_hash = hash(file_info_tuple)
 
            # Include all parameters that affect output in the hash string
            return f"{file_list_hash}_{n_images}_{sort}_{export_with_alpha}_{seed if not sort else 'sorted'}"
        except Exception as e:
             print(f"sokes_nodes.py IS_CHANGED Error: {e}")
             return float("NaN")
  
# END: Load Random Image with Path and Mask | Sokes 🦬
##############################################################

# Unused utility functions (can be kept or removed)
def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))
def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]; target_aspect = w/h
    if target_aspect > aspect_ratio: new_w = int(h * aspect_ratio); left = (w - new_w) // 2; crop = img[:, left:left+new_w]
    else: new_h = int(w / aspect_ratio); top = (h - new_h) // 2; crop = img[top:top+new_h, :]
    return crop
def create_same_sized_crops(imgs, target_n_pixels=2048**2):
    imgs_np = [np.array(img) for img in imgs]; aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs_np]
    final_aspect_ratio = np.mean(aspect_ratios); crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs_np]
    final_h = np.sqrt(target_n_pixels / final_aspect_ratio); final_w = final_h * final_aspect_ratio
    final_h = round_to_nearest_multiple(final_h, 8); final_w = round_to_nearest_multiple(final_w, 8)
    return [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]

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
MANIFEST = {
    "name": "Sokes Nodes", "version": (1,0,2), "author": "Sokes",
    "project": "https://github.com/Sokes/ComfyUI-Sokes",
    "description": "A collection of utility nodes for ComfyUI by Sokes.",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'MANIFEST']
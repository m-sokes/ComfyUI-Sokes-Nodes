from datetime import datetime # datetime
import re # regex
import os # for load random image
import random # for load random image
import imghdr # to check image by header vs extension
import torch # to resize images
import numpy as np # for image manipulation in torch
import cv2 # for image processing
from PIL import Image, ImageOps # load random image, orient image

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


##############################################################
# START: Latent Input Swtich x9 | Sokes 🦬

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

# END: Latent Input Swtich x9 | Sokes 🦬
##############################################################

##############################################################
# START: Replace Text with Regex | Sokes 🦬

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


# END: Replace Text with Regex | Sokes 🦬
##############################################################

##############################################################
# START: Load Random Image with Path and Mask | Sokes 🦬
 
class load_random_image_sokes:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG"]
 
    @classmethod
    def INPUT_TYPES(s):
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
        # --- Folder Resolution (same as before) ---
        if not os.path.isdir(folder):
            try:
                input_dir = folder_paths.get_input_directory()
                resolved_folder = folder_paths.get_annotated_filepath(folder) # More robust way
                if resolved_folder and os.path.isdir(resolved_folder):
                     folder = resolved_folder
                elif os.path.isdir(os.path.join(input_dir, folder)): # Fallback for simple relative paths
                    folder = os.path.join(input_dir, folder)
                else:
                     if not os.path.isdir(folder):
                          raise FileNotFoundError(f"Folder '{folder}' not found directly or relative to input directory.")
            except Exception as e:
                 raise Exception(f"Error resolving folder '{folder}'. Ensure it exists or is relative to ComfyUI's input directory. Error: {e}")
 
        # --- File Discovery (same as before) ---
        try:
            all_files = [os.path.join(folder, f) for f in os.listdir(folder)]
            image_paths = [f for f in all_files if os.path.isfile(f)]
            image_paths = [f for f in image_paths if any([f.lower().endswith(ext) for ext in self.img_extensions])]
        except Exception as e:
            raise Exception(f"Error listing files in folder '{folder}': {e}")
 
        # --- Image Validation (same as before) ---
        valid_image_paths = []
        for f in image_paths:
            try:
                is_image = imghdr.what(f)
                if not is_image:
                    try:
                        img_test = Image.open(f)
                        img_test.verify()
                        is_image = True
                    except Exception:
                        is_image = False
                if is_image:
                    valid_image_paths.append(f)
            except Exception as e:
                 print(f"Skipping potentially corrupt image: {f} - {str(e)}")
 
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")
 
        # --- Shuffle/Sort and Select (same as before) ---
        if not sort:
            random.seed(seed)
            random.shuffle(valid_image_paths)
        else:
             valid_image_paths = sorted(valid_image_paths)
 
        num_available = len(valid_image_paths)
        actual_n_images = min(n_images, num_available)
        if actual_n_images < n_images and n_images > 0:
             print(f"Warning: Requested {n_images} images, but only {num_available} were found/valid in '{folder}'. Loading {actual_n_images}.")
        elif n_images <= 0:
             actual_n_images = num_available
 
        selected_paths = valid_image_paths[:actual_n_images]
 
        # --- Load and Process Images and Masks ---
        output_images = []
        output_masks = []
        loaded_paths = []
        first_image_shape = None
 
        for image_path in selected_paths:
            try:
                img_pil = Image.open(image_path)
                img_pil = ImageOps.exif_transpose(img_pil) # Handle orientation
 
                # --- Image Processing ---
                # **MODIFICATION: Convert to RGBA to preserve/add alpha channel**
                image_rgba = img_pil.convert("RGBA")
                image_np = np.array(image_rgba).astype(np.float32) / 255.0 # Shape: (H, W, 4)
                # ComfyUI IMAGE format is Batch, Height, Width, Channel (BHWC)
                image_tensor = torch.from_numpy(image_np)[None,] # Shape: [1, H, W, 4]
 
                # --- Mask Processing (Remains the same) ---
                # Checks the *original* PIL image for alpha before RGBA conversion
                mask_tensor = None
                # ComfyUI MASK format is Batch, Height, Width (BHW)
                if 'A' in img_pil.getbands() or 'a' in img_pil.getbands(): # Check original bands
                    mask = img_pil.getchannel('A')
                    mask_np = np.array(mask).astype(np.float32) / 255.0 # Shape: (H, W)
                    mask_tensor = torch.from_numpy(mask_np)[None,] # Shape: [1, H, W]
                else:
                    mask_shape = (image_tensor.shape[1], image_tensor.shape[2]) # Get H, W
                    mask_np = np.ones(mask_shape, dtype=np.float32)
                    mask_tensor = torch.from_numpy(mask_np)[None,] # Shape: [1, H, W]
 
                # --- Batch Consistency Check (Checks H, W - still valid) ---
                current_shape = image_tensor.shape[1:3]
                if first_image_shape is None:
                    first_image_shape = current_shape
                elif current_shape != first_image_shape:
                    print(f"⚠️ Warning: Image {os.path.basename(image_path)} has different dimensions ({current_shape}) than the first image ({first_image_shape}). Batching may fail or produce unexpected results downstream. Consider resizing images beforehand.")
                    # Optional resizing logic could go here, applied to both image_tensor (4ch) and mask_tensor (1ch)
 
                output_images.append(image_tensor)
                output_masks.append(mask_tensor)
                loaded_paths.append(image_path)
 
            except Exception as e:
                print(f"❌ Error loading or processing image {image_path}: {str(e)}. Skipping.")
                continue
 
        if not output_images:
            raise ValueError("No images were successfully loaded after processing.")
 
        # --- Collate Batches (same as before) ---
        final_image_batch = torch.cat(output_images, dim=0) # Shape: [B, H, W, 4]
        final_mask_batch = torch.cat(output_masks, dim=0)  # Shape: [B, H, W]
 
        return (final_image_batch, final_mask_batch, loaded_paths)
 
    # --- IS_CHANGED (same as before) ---
    @classmethod
    def IS_CHANGED(s, folder, n_images, seed, sort):
        try:
            resolved_folder = None
            if not os.path.isdir(folder):
                try:
                    resolved_folder = folder_paths.get_annotated_filepath(folder)
                    if not resolved_folder or not os.path.isdir(resolved_folder):
                       input_dir = folder_paths.get_input_directory()
                       resolved_folder = os.path.join(input_dir, folder)
                       if not os.path.isdir(resolved_folder):
                            resolved_folder = folder
                except:
                    resolved_folder = folder
 
            if not os.path.isdir(resolved_folder):
                 return float("NaN")
 
            all_files = [os.path.join(resolved_folder, f) for f in os.listdir(resolved_folder)]
            image_paths = [f for f in all_files if os.path.isfile(f)]
            img_extensions_lower = [ext.lower() for ext in s().img_extensions]
            image_paths = [f for f in image_paths if f.lower().endswith(tuple(img_extensions_lower))]
            image_paths = sorted(image_paths)
            mtimes = [os.path.getmtime(f) for f in image_paths]
            file_info_tuple = tuple(zip(image_paths, mtimes))
            file_list_hash = hash(file_info_tuple)
 
            if not sort:
                return f"{file_list_hash}_{seed}_{n_images}"
            else:
                return f"{file_list_hash}_{n_images}"
 
        except Exception as e:
             print(f"IS_CHANGED Error: {e}")
             return float("NaN")
  
# END: Load Random Image with Path and Mask | Sokes 🦬
##############################################################


def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))


def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]
    if w/h > aspect_ratio:
        # crop width:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        crop = img[:, left:right]
    else:
        # crop height:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2
        crop = img[top:bottom, :]
    return crop


def create_same_sized_crops(imgs, target_n_pixels=2048**2):
    """
    Given a list of images:
        - extract the best possible center crop of same aspect ratio for all images
        - rescale these crops to have ~target_n_pixels
        - return resized images
    """

    assert len(imgs) > 1
    imgs = [np.array(img) for img in imgs]
    
    # Get center crops at same aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs]
    final_aspect_ratio = np.mean(aspect_ratios)
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs]

    # Compute final w,h using final_aspect_ratio and target_n_pixels:
    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = final_h * final_aspect_ratio
    final_h = round_to_nearest_multiple(final_h, 8)
    final_w = round_to_nearest_multiple(final_w, 8)

    # Resize images
    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    #resized_imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in resized_imgs]
    
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

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

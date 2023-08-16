import datetime

# Custom Date Format | Sokes 🦬

class custom_date_format_sokes:
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
    FUNCTION = "custom_date_format_sokes"

    CATEGORY = "sokes/Date_Time"
    
    def custom_date_format_sokes(self, date_format):
        today_date = datetime.date.today()
        customDateFormat = date_format.upper()
        customDateFormat = customDateFormat.replace("YYYY", today_date.strftime("%Y"))
        customDateFormat = customDateFormat.replace("MM", today_date.strftime("%m"))
        customDateFormat = customDateFormat.replace("M", today_date.strftime("%m").lstrip("0").replace(" 0", " "))
        customDateFormat = customDateFormat.replace("DD", today_date.strftime("%d"))
        customDateFormat = customDateFormat.replace("D", today_date.strftime("%d").lstrip("0").replace(" 0", " "))
        return (customDateFormat, )


# Latent Input Swtich x9 | Sokes 🦬

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
    CATEGORY = "sokes/Latent"

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


NODE_CLASS_MAPPINGS = {
    "Latent Switch x9 | sokes 🦬": latent_input_switch_9x_sokes,
    "Custom Date Format | sokes 🦬": custom_date_format_sokes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Custom Date Format | sokes 🦬": "Custom Date Format 🦬",
    "Latent Switch x9 | sokes 🦬": "Latent Switch x9 🦬",
}
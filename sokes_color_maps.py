# sokes_color_maps.py
import webcolors

# Try to get the CSS3 names-to-hex mapping from webcolors.
try:
    css3_names_to_hex = webcolors.css3_names_to_hex
except AttributeError:
    try:
        css3_names_to_hex = webcolors.CSS3_NAMES_TO_HEX
    except AttributeError:
        # Fallback mapping in case neither attribute exists.
        css3_names_to_hex = {
            'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7', 'aqua': '#00ffff',
            'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc',
            'bisque': '#ffe4c4', 'black': '#000000', 'blanchedalmond': '#ffebcd',
            'blue': '#0000ff', 'blueviolet': '#8a2be2', 'brown': '#a52a2a',
            'burlywood': '#deb887', 'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00',
            'chocolate': '#d2691e', 'coral': '#ff7f50', 'cornflowerblue': '#6495ed',
            'cornsilk': '#fff8dc', 'crimson': '#dc143c', 'cyan': '#00ffff',
            'darkblue': '#00008b', 'darkcyan': '#008b8b', 'darkgoldenrod': '#b8860b',
            # NOTE: Corrected 'darkgrey' and 'dimgrey' keys if using standard names
            'darkgray': '#a9a9a9', # Alias for darkgrey often needed
            'darkgrey': '#a9a9a9',
            'darkgreen': '#006400', 'darkkhaki': '#bdb76b', 'darkmagenta': '#8b008b',
            'darkolivegreen': '#556b2f', 'darkorange': '#ff8c00', 'darkorchid': '#9932cc',
            'darkred': '#8b0000', 'darksalmon': '#e9967a', 'darkseagreen': '#8fbc8f',
            'darkslateblue': '#483d8b',
            'darkslategray': '#2f4f4f', # Alias for darkslategrey often needed
            'darkslategrey': '#2f4f4f',
            'darkturquoise': '#00ced1', 'darkviolet': '#9400d3', 'deeppink': '#ff1493',
            'deepskyblue': '#00bfff',
            'dimgray': '#696969', # Alias for dimgrey often needed
            'dimgrey': '#696969',
            'dodgerblue': '#1e90ff', 'firebrick': '#b22222', 'floralwhite': '#fffaf0',
            'forestgreen': '#228b22', 'fuchsia': '#ff00ff', 'gainsboro': '#dcdcdc',
            'ghostwhite': '#f8f8ff', 'gold': '#ffd700', 'goldenrod': '#daa520',
            'gray': '#808080', # Alias for grey often needed
            'grey': '#808080',
            'green': '#008000', 'greenyellow': '#adff2f', 'honeydew': '#f0fff0',
            'hotpink': '#ff69b4', 'indianred': '#cd5c5c', 'indigo': '#4b0082',
            'ivory': '#fffff0', 'khaki': '#f0e68c', 'lavender': '#e6e6fa',
            'lavenderblush': '#fff0f5', 'lawngreen': '#7cfc00', 'lemonchiffon': '#fffacd',
            'lightblue': '#add8e6', 'lightcoral': '#f08080', 'lightcyan': '#e0ffff',
            'lightgoldenrodyellow': '#fafad2',
            'lightgray': '#d3d3d3', # Alias for lightgrey often needed
            'lightgrey': '#d3d3d3',
            'lightgreen': '#90ee90', 'lightpink': '#ffb6c1', 'lightsalmon': '#ffa07a',
            'lightseagreen': '#20b2aa', 'lightskyblue': '#87cefa',
            'lightslategray': '#778899', # Alias for lightslategrey often needed
            'lightslategrey': '#778899',
            'lightsteelblue': '#b0c4de', 'lightyellow': '#ffffe0', 'lime': '#00ff00',
            'limegreen': '#32cd32', 'linen': '#faf0e6', 'magenta': '#ff00ff',
            'maroon': '#800000', 'mediumaquamarine': '#66cdaa', 'mediumblue': '#0000cd',
            'mediumorchid': '#ba55d3', 'mediumpurple': '#9370db', 'mediumseagreen': '#3cb371',
            'mediumslateblue': '#7b68ee', 'mediumspringgreen': '#00fa9a', 'mediumturquoise': '#48d1cc',
            'mediumvioletred': '#c71585', 'midnightblue': '#191970', 'mintcream': '#f5fffa',
            'mistyrose': '#ffe4e1', 'moccasin': '#ffe4b5', 'navajowhite': '#ffdead',
            'navy': '#000080', 'oldlace': '#fdf5e6', 'olive': '#808000',
            'olivedrab': '#6b8e23', 'orange': '#ffa500', 'orangered': '#ff4500',
            'orchid': '#da70d6', 'palegoldenrod': '#eee8aa', 'palegreen': '#98fb98',
            'paleturquoise': '#afeeee', 'palevioletred': '#db7093', 'papayawhip': '#ffefd5',
            'peachpuff': '#ffdab9', 'peru': '#cd853f', 'pink': '#ffc0cb',
            'plum': '#dda0dd', 'powderblue': '#b0e0e6', 'purple': '#800080',
            'rebeccapurple': '#663399', 'red': '#ff0000', 'rosybrown': '#bc8f8f',
            'royalblue': '#4169e1', 'saddlebrown': '#8b4513', 'salmon': '#fa8072',
            'sandybrown': '#f4a460', 'seagreen': '#2e8b57', 'seashell': '#fff5ee',
            'sienna': '#a0522d', 'silver': '#c0c0c0', 'skyblue': '#87ceeb',
            'slateblue': '#6a5acd',
            'slategray': '#708090', # Alias for slategrey often needed
            'slategrey': '#708090',
            'snow': '#fffafa', 'springgreen': '#00ff7f', 'steelblue': '#4682b4',
            'tan': '#d2b48c', 'teal': '#008080', 'thistle': '#d8bfd8',
            'tomato': '#ff6347', 'turquoise': '#40e0d0', 'violet': '#ee82ee',
            'wheat': '#f5deb3', 'white': '#ffffff', 'whitesmoke': '#f5f5f5',
            'yellow': '#ffff00', 'yellowgreen': '#9acd32',
             # Removed duplicates from your original map if they existed
            # 'paleyellow': '#e1e34f',
            # 'paleorange': '#e39e4f'
        }
# Add any missing CSS3 colors if needed
if 'darkgray' not in css3_names_to_hex: css3_names_to_hex['gray'] = '#a9a9a9'
if 'dimgray' not in css3_names_to_hex: css3_names_to_hex['gray'] = '#696969'
# ... add others if webcolors version is very old ...

# Reverse the mapping to obtain a hex-to-name dict (ensure unique hex values)
# Handle potential duplicate hex values (like aqua/cyan) by preferring one name
css3_hex_to_names = {}
for name, hex_code in css3_names_to_hex.items():
    if hex_code not in css3_hex_to_names: # Keep the first name encountered for a hex
        css3_hex_to_names[hex_code] = name
# Ensure essential colors are present if duplicates overwrote them
if '#00ffff' not in css3_hex_to_names: css3_hex_to_names['#00ffff'] = 'aqua' # Or 'cyan'
if '#ff00ff' not in css3_hex_to_names: css3_hex_to_names['#ff00ff'] = 'fuchsia' # Or 'magenta'
# ... Add more specific checks if needed


# <<< ADD THIS SECTION >>>
# --- Explicit Targets for Priority Comparison ---
# Internal Name -> Hex Code
# These are checked first before iterating through all CSS3 colors.
# The internal name should have a corresponding entry in human_readable_map below
# if you want a specific output name for it.
explicit_targets_for_comparison = {
    # custom colors: "friendly name here": "#hex"
    #RED
    "dark pink": "#8B008B",
    "pink": "#F5B4FE", #was lavender
    "neon pink": "#FF1F62", #was deep red
    "dark red": "#180101", #was black
    "warm brown": "#251313", #was dark red
    "muted pale red": "#A05A5A", #was terracotta
    "muted red": "#BD1436", #was muted brown
    #ORANGE
    "orange": "#FFA257", #was tan
    "bright orange-red": "#FD3E28", #was bright red
    "terracotta": "#cb6843", #was rich brown
    #YELLOW
    "pale yellow": "#FEFFAD",
    "neon yellow": "#FFEA00", #neon yellow
    #BROWN
    "dark brown": "#1D1507", #dark brown
    "dark brown": "#3D2B1F", # Target for dark browns
    "dark brown": "#1A1305", #was dark gray
    "brown": "#2E1000", #was dark red
    #GREEN
    "muted dark green": "#003832", #was charcoal gray
    "green": "#11AC1B", #was neon green
    "muted greenish-brown": "#3D3E28", #was medium gray
    "pale muted green": "#535F58", #muted green
    "dark green": "#011E03", #dark green
    "muted green": "#293829", #was muted greenish-brown
    "neon chartreuse": "#D4FF00", #was chartreuse
    #BLUE
    "very dark blue": "#01002E", #was dark blue
    #PURPLE
    "bright purple": "#A600FF", # Target for bright purples
    "dark purple": "#29002E", #was very dark blue
    "dark purple": "#140222", #was very dark blue
    #GRAYs
    "white": "#FFFFFF",          # Standard extreme
    "gray": "#4F4F4F", #gray, cause pale muted green takes overs
    "dark gray": "#1C1C1C", #dark gray
    "black": "#000000",          # Standard extreme
    # Add more specific targets here if desired, e.g.:
    # "customforestgreen": "#014421",
    # END CUSTOM COLORS
}

# --- Human Readable Map ---
# Internal Name (CSS3 or Custom) -> User-Facing Name
human_readable_map = {
    # --- ADD MAPPINGS FOR YOUR CUSTOM TARGET NAMES ---
    'customdarkbrown': 'dark brown',
    # 'customforestgreen': 'deep forest green', # Example if added above

    # --- YOUR EXISTING MAPPINGS BELOW ---
    'aliceblue': 'pale sky blue',
    'antiquewhite': 'warm off-white',
    'aqua': 'bright cyan', # Note: #00ffff might map to 'cyan' depending on dict order
    'aquamarine': 'teal',
    'azure': 'light cyan',
    'beige': 'muted tan',
    'bisque': 'pale orange',
    'black': 'black', # Explicitly map black
    'blanchedalmond': 'pale peach',
    'blue': 'bright blue',
    'blueviolet': 'purple',
    'brown': 'muted brown',
    'burlywood': 'sandy brown',
    'cadetblue': 'slate blue',
    'chartreuse': 'neon green',
    'chocolate': 'rich brown',
    'coral': 'salmon pink',
    'cornflowerblue': 'soft periwinkle',
    'cornsilk': 'pale yellow', # Changed from 'pale cream' in your original map
    'crimson': 'bright red',
    'cyan': 'bright cyan', # Note: #00ffff might map to 'aqua' depending on dict order
    'darkblue': 'dark blue', # Changed from 'deep navy' in your original map
    'darkcyan': 'muted teal',
    'darkgoldenrod': 'mustard',
    'darkgray': 'medium gray', # Use 'darkgray' consistently
    'darkgrey': 'medium gray', # Keep alias just in case
    'darkgreen': 'dark green', # Changed from 'forest green'
    'darkkhaki': 'chartreuse',
    'darkmagenta': 'deep purple', # This is the key one for you
    'darkolivegreen': 'green',
    'darkorange': 'bright orange',
    'darkorchid': 'deep lavender',
    'darkred': 'burgundy',
    'darksalmon': 'pale pink',
    'darkseagreen': 'sage green',
    'darkslateblue': 'slate blue',
    'darkslategray': 'charcoal gray', # Use 'darkslategray' consistently
    'darkslategrey': 'charcoal gray', # Keep alias
    'darkturquoise': 'bright teal',
    'darkviolet': 'deep violet',
    'deeppink': 'neon pink',
    'deepskyblue': 'pale blue',
    'dimgray': 'stone gray', # Use 'dimgray' consistently
    'dimgrey': 'stone gray', # Keep alias
    'dodgerblue': 'bright azure',
    'firebrick': 'bright red',
    'floralwhite': 'tan',
    'forestgreen': 'muted green',
    'fuchsia': 'neon pink', # Note: #ff00ff might map to 'magenta'
    'gainsboro': 'silver',
    'ghostwhite': 'cool white',
    'gold': 'gold',
    'goldenrod': 'ochre',
    'gray': 'medium gray', # Use 'gray' consistently
    'grey': 'medium gray', # Keep alias
    'green': 'green',
    'greenyellow': 'chartreuse',
    'honeydew': 'pale mint',
    'hotpink': 'bright pink',
    'indianred': 'terracotta',
    'indigo': 'royal purple', # Keep this as royal purple
    'ivory': 'warm white',
    'khaki': 'tan',
    'lavender': 'pale lilac',
    'lavenderblush': 'light purple',
    'lawngreen': 'neon green',
    'lemonchiffon': 'pale yellow',
    'lightblue': 'powder blue',
    'lightcoral': 'soft coral color',
    'lightcyan': 'ice blue',
    'lightgoldenrodyellow': 'pale gold',
    'lightgray': 'silver', # Use 'lightgray' consistently
    'lightgrey': 'silver', # Keep alias
    'lightgreen': 'pale green',
    'lightpink': 'light pink',
    'lightsalmon': 'peach',
    'lightseagreen': 'bright teal',
    'lightskyblue': 'light blue',
    'lightslategray': 'light pale blue', # Use 'lightslategray' consistently
    'lightslategrey': 'light gray', # Keep alias
    'lightsteelblue': 'pale blue',
    'lightyellow': 'pale yellow', # Changed from 'pale cream'
    'lime': 'neon green',
    'limegreen': 'neon green',
    'linen': 'natural linen',
    'magenta': 'bright pink', # Note: #ff00ff might map to 'fuchsia'
    'maroon': 'dark red',
    'mediumaquamarine': 'sea green',
    'mediumblue': 'bright blue',
    'mediumorchid': 'soft purple',
    'mediumpurple': 'purple',
    'mediumseagreen': 'sage green',
    'mediumslateblue': 'periwinkle',
    'mediumspringgreen': 'bright green',
    'mediumturquoise': 'teal',
    'mediumvioletred': 'raspberry',
    'midnightblue': 'dark blue', # Changed from 'navy blue'
    'mintcream': 'pale green',
    'mistyrose': 'very light pink',
    'moccasin': 'light tan',
    'navajowhite': 'pale orange',
    'navy': 'dark navy',
    'oldlace': 'light tan',
    'olive': 'army green',
    'olivedrab': 'khaki green',
    'orange': 'bright orange',
    'orangered': 'vibrant red-orange',
    'orchid': 'purple',
    'palegoldenrod': 'pale yellow',
    'palegreen': 'pastel green',
    'paleturquoise': 'light neon blue',
    'palevioletred': 'pink',
    'papayawhip': 'pale orange', # Changed from 'creamy peach'
    'peachpuff': 'soft peach',
    'peru': 'terracotta',
    'pink': 'pale pink',
    'plum': 'pale purple',
    'powderblue': 'soft blue',
    'purple': 'royal purple',
    'rebeccapurple': 'purple',
    'red': 'bright red',
    'rosybrown': 'pink',
    'royalblue': 'cobalt blue',
    'saddlebrown': 'brown',
    'salmon': 'coral',
    'sandybrown': 'tan', # Changed from 'sand'
    'seagreen': 'sea green',
    'seashell': 'pale blush',
    'sienna': 'burnt sienna',
    'silver': 'metallic silver',
    'skyblue': 'bright sky blue',
    'slateblue': 'muted periwinkle',
    'slategray': 'steel gray', # Use 'slategray' consistently
    'slategrey': 'steel gray', # Keep alias
    'snow': 'bright white',
    'springgreen': 'lime',
    'steelblue': 'slate blue',
    'tan': 'tan', # Changed from 'sand'
    'teal': 'bright cyan', # Changed from 'bright teal'
    'thistle': 'pale lilac',
    'tomato': 'soft red',
    'turquoise': 'neon turquoise',
    'violet': 'lavender',
    'wheat': 'pale tan',
    'white': 'pure white',
    'whitesmoke': 'off-white',
    'yellow': 'bright yellow',
    'yellowgreen': 'muted lime-green',
    # Your originals below:
    'paleyellow': 'pale yellow',
    'paleorange': 'pale orange'
}

"""
======================================================================
 Title:                   ASDA - Automated Spot Detection & Anamysis
 Creating Author:         Janan ARSLAN
 Creation Date:           [27-11-2024]
 Latest Modification:     [20-08-32025]
 Modification Author:     Janan ARSLAN
 E-mail:                  janan.arslan@gmail.com
 Version:                 v1.10
======================================================================


"""

"""Configuration settings and color definitions."""

import numpy as np

# Analysis constants
BINARY_THRESHOLD = 20
CONNECTIVITY = 4
DRAW_CIRCLE_RADIUS = 4
DEFAULT_MIN_AREA = 20

# Color fallback mappings
COLOR_FALLBACKS = {
    'turquoise': 'light_blue',
    'dark_blue': 'light_blue'
}

# BGR colors for visualization
BGR_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'turquoise': (255, 200, 0),
    'light_blue': (255, 150, 0),
    'magenta': (255, 0, 255),
    'light_green': (100, 255, 100),
    'dark_gray': (100, 100, 100),
    'light_gray': (200, 200, 200),
    'dark_blue': (200, 0, 0)
}

# DBSCAN epsilon values for different colors
DBSCAN_EPS = {
    'dark_blue': 30,
    'light_gray': 35,
    'default': 25
}

def get_hsv_color_ranges():
    """Returns dictionary of color names to HSV ranges."""
    
    # Standard color definitions
    colors = {
        'red1': {'lower': [0, 120, 100], 'upper': [10, 255, 255]},
        'red2': {'lower': [170, 120, 100], 'upper': [180, 255, 255]},
        'yellow': {'lower': [25, 150, 100], 'upper': [35, 255, 255]},
        'orange': {'lower': [15, 150, 100], 'upper': [24, 255, 255]},
        'turquoise': {'lower': [85, 100, 100], 'upper': [95, 255, 255], 'max_area': 500, 'min_area': 80},
        'light_blue': {'lower': [95, 50, 50], 'upper': [105, 255, 255], 'special_process': True, 'min_area': 80},
        'magenta': {'lower': [140, 100, 100], 'upper': [160, 255, 255]},
        'light_green': {'lower': [58, 90, 190], 'upper': [68, 115, 220]},
        'dark_gray': {'lower': [0, 0, 50], 'upper': [180, 30, 130], 'special_process': True, 'exclude_overlap': True},
        'light_gray': {'lower': [0, 0, 130], 'upper': [180, 40, 230], 'exclude_overlap': True, 'max_area': 600},
        'dark_blue': {'lower': [105, 120, 80], 'upper': [125, 255, 255], 'exclude_overlap': True}
    }

    # Polychrome colors (converted from 360-degree hue to 180-degree)
    polychrome = {
        'dark_gray_poly': {'lower': [0, 10, 30], 'upper': [180, 30, 70]},
        'light_gray_poly': {'lower': [150, 0, 85], 'upper': [180, 20, 95]},
        'bright_red_poly': {'lower': [177, 85, 85], 'upper': [180, 100, 100]},
        'bright_magenta_poly': {'lower': [150, 90, 90], 'upper': [151, 100, 100]},
        'bright_green_poly': {'lower': [125, 90, 90], 'upper': [130, 100, 100]},
        'bright_blue_poly': {'lower': [107, 80, 90], 'upper': [112, 100, 100]},
        'orange_poly': {'lower': [35, 90, 90], 'upper': [45, 100, 100]},
        'deep_red_poly': {'lower': [165, 90, 60], 'upper': [170, 100, 70]},
        'turquoise_poly': {'lower': [165, 85, 85], 'upper': [170, 100, 100]},
        'olive_poly': {'lower': [70, 70, 50], 'upper': [80, 90, 70]},
        'cyan_poly': {'lower': [95, 80, 85], 'upper': [100, 100, 100]},
        'navy_poly': {'lower': [110, 50, 40], 'upper': [115, 70, 65]},
        'brown_poly': {'lower': [15, 70, 60], 'upper': [25, 90, 80]},
        'forest_green_poly': {'lower': [150, 65, 35], 'upper': [160, 85, 55]},
        'dark_olive_poly': {'lower': [45, 80, 35], 'upper': [55, 100, 55]},
        'dark_magenta_poly': {'lower': [155, 85, 60], 'upper': [160, 100, 70]},
        'yellow_poly': {'lower': [50, 80, 85], 'upper': [60, 100, 100]},
        'mint_poly': {'lower': [140, 70, 60], 'upper': [150, 90, 80]},
        'pink_poly': {'lower': [165, 90, 85], 'upper': [170, 100, 100]},
        'bright_magenta2_poly': {'lower': [160, 85, 85], 'upper': [165, 100, 100]},
        'pale_yellow_poly': {'lower': [45, 30, 85], 'upper': [55, 50, 100]},
        'pale_blue_poly': {'lower': [112, 25, 85], 'upper': [117, 45, 100]},
        'teal_poly': {'lower': [175, 40, 75], 'upper': [180, 60, 90]},
        'dark_teal_poly': {'lower': [95, 60, 45], 'upper': [100, 80, 65]}
    }

    # Add exclude_overlap=True to all polychrome colors
    for poly_color in polychrome.values():
        poly_color['exclude_overlap'] = True

    # Merge all colors
    colors.update(polychrome)
    return colors

def parse_base_color(color_name):
    """Extract base color from compound names."""
    parts = color_name.split("_")
    if len(parts) >= 2:
        two_part = "_".join(parts[:2]).lower()
        if two_part in BGR_COLORS:
            return two_part
    return parts[0].lower()

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

"""Image processing and preprocessing functions."""

import numpy as np
import cv2
from skimage import io as skimage_io
from config import BINARY_THRESHOLD, CONNECTIVITY

def load_image(image_path):
    """Load and convert image to RGB format."""
    img = skimage_io.imread(image_path)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    elif img.shape[2] == 4:
        print("Converting RGBA to RGB...")
        img = img[:, :, :3]
    return img

def get_key_region_mask(image_shape, buffer=50, fraction=0.25):
    """Create mask for key region (rightmost portion of image)."""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    key_width = int(width * fraction)
    start_col = max(0, width - key_width - buffer)
    mask[:, start_col:] = 255
    return mask

def find_largest_square(image):
    """Find the largest square bounding box in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 4)
    best_area = 0
    best_bbox = None
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > best_area:
            best_area = area
            best_bbox = (x, y, w, h)
    
    if best_bbox is None:
        return None
    
    x, y, w, h = best_bbox
    side = max(w, h)
    return (x, y, x + side, y + side)

def crop_image(image, bbox):
    """Crop image to bounding box."""
    if bbox is None:
        return image
    
    x_min, y_min, x_max, y_max = bbox
    h, w = image.shape[:2]
    x_min = max(0, min(x_min, w))
    x_max = max(0, min(x_max, w))
    y_min = max(0, min(y_min, h))
    y_max = max(0, min(y_max, h))
    
    return image[y_min:y_max, x_min:x_max]

def extract_foreground(image):
    """Extract foreground spots using edge detection."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Edge detection and enhancement
    laplacian = cv2.Laplacian(gray, cv2.CV_8UC1)
    dilated = cv2.dilate(laplacian, np.ones((5, 5), np.uint8))
    _, thresh = cv2.threshold(dilated, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Create mask from detected components
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)
    mask = np.zeros_like(gray)

    for i in range(1, num_labels):
        radius = int(np.sqrt(1000 / np.pi))
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)

    # Apply mask to original image
    foreground = cv2.bitwise_and(bgr, bgr, mask=mask)
    background = np.zeros_like(bgr)
    result = cv2.copyTo(foreground, mask, background)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def adjust_injection_site_for_crop(injection_site, bbox):
    """Adjust injection site coordinates for cropped image."""
    if bbox is None:
        return injection_site
    
    x_min, y_min, _, _ = bbox
    adjusted_x = injection_site[0] - x_min
    adjusted_y = injection_site[1] - y_min
    
    return (adjusted_x, adjusted_y)

def create_edge_mask(image, fraction=0.25, buffer=50):
    """Create edge-based mask for spot detection."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    binary_image = cv2.Laplacian(gray, cv2.CV_8UC1)
    dilated_image = cv2.dilate(binary_image, np.ones((5, 5), np.uint8))
    _, thresh = cv2.threshold(dilated_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Exclude key region if needed
    if fraction > 0:
        key_mask = get_key_region_mask(image.shape, buffer=buffer, fraction=fraction)
        main_mask = cv2.bitwise_not(key_mask)
        thresh = cv2.bitwise_and(thresh, thresh, mask=main_mask)

    # Draw circles for consistency
    components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)
    centers = components[3]
    for center in centers:
        cv2.circle(thresh, (int(center[0]), int(center[1])), 4, (255), thickness=-1)

    return thresh

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

"""Spot detection algorithms and analysis functions."""

import math
import numpy as np
import pandas as pd
import cv2
from copy import deepcopy
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance

from config import get_hsv_color_ranges, COLOR_FALLBACKS, BGR_COLORS, DBSCAN_EPS, DEFAULT_MIN_AREA
from image_processing import get_key_region_mask, create_edge_mask

class SpotProperties:
    """Properties of a detected spot."""
    
    def __init__(self, y, x, image, area):
        self.centroid = (y, x)
        self.area = area
        # Extract small patch around spot for intensity analysis
        y_min = max(0, y - 6)
        y_max = min(image.shape[0], y + 7)
        x_min = max(0, x - 6)
        x_max = min(image.shape[1], x + 7)
        self.intensity_image = image[y_min:y_max, x_min:x_max]

def detect_key_colors(image, fraction=0.25, buffer=50):
    """Detect which colors are present in the key region."""
    key_mask = get_key_region_mask(image.shape, buffer=buffer, fraction=fraction)
    bgr = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2BGR)
    bgr[key_mask == 0] = 0
    key_only_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    spots_in_key = detect_color_spots(key_only_rgb, fraction=0.0, buffer=0)
    return set(spots_in_key.keys())

def detect_color_spots(image, fraction=0.25, buffer=50):
    """Detect colored spots in the image."""
    color_dict = get_hsv_color_ranges()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Create edge-based threshold mask
    thresh = create_edge_mask(image, fraction, buffer)
    
    spots_by_color = {}
    used_positions_global = set()
    used_positions_by_color = {}

    # Handle red1+red2 combination first
    if 'red1' in color_dict and 'red2' in color_dict:
        red_spots = _detect_red_spots(hsv, thresh, image, color_dict)
        if red_spots:
            unified = _unify_close_spots(red_spots, image, 'red')
            spots_by_color['red'] = unified
            used_positions_by_color['red'] = set(sp.centroid for sp in unified)
            for sp in unified:
                used_positions_global.add(sp.centroid)
        else:
            used_positions_by_color['red'] = set()

    # Process other colors
    for color_name, color_range in color_dict.items():
        if color_name in ['red1', 'red2']:
            continue
            
        spots = _detect_single_color(
            hsv, thresh, image, color_name, color_range,
            used_positions_global, used_positions_by_color
        )
        
        if spots:
            unified = _unify_close_spots(spots, image, color_name)
            spots_by_color[color_name] = unified
            used_positions_by_color[color_name] = set(sp.centroid for sp in unified)
            for sp in unified:
                used_positions_global.add(sp.centroid)
        else:
            spots_by_color[color_name] = []
            used_positions_by_color[color_name] = set()

    return spots_by_color

def _detect_red_spots(hsv, thresh, image, color_dict):
    """Detect red spots (combines red1 and red2 ranges)."""
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(color_dict['red1']['lower']), np.array(color_dict['red1']['upper'])),
        cv2.inRange(hsv, np.array(color_dict['red2']['lower']), np.array(color_dict['red2']['upper']))
    )
    red_mask = cv2.bitwise_and(red_mask, thresh)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask, 4, cv2.CV_32S)
    red_spots = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > DEFAULT_MIN_AREA:
            y, x = int(centroids[i][1]), int(centroids[i][0])
            spot = SpotProperties(y, x, image, area)
            red_spots.append(spot)
    
    return red_spots

def _detect_single_color(hsv, thresh, image, color_name, color_range, 
                        used_positions_global, used_positions_by_color):
    """Detect spots of a single color."""
    lower = np.array(color_range['lower'], dtype=np.uint8)
    upper = np.array(color_range['upper'], dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lower, upper)
    color_mask = cv2.bitwise_and(color_mask, thresh)

    # Apply special processing if needed
    if color_range.get('special_process'):
        color_mask = _apply_special_processing(hsv, color_mask, color_name)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(color_mask, 4, cv2.CV_32S)
    
    # Get parameters for this color
    global_exclude_dist = color_range.get('exclude_distance', 10)
    same_color_dist = 15
    max_area = color_range.get('max_area', None)
    min_area = color_range.get('min_area', DEFAULT_MIN_AREA)
    
    spots = []
    if color_name not in used_positions_by_color:
        used_positions_by_color[color_name] = set()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or (max_area and area > max_area):
            continue

        y, x = int(centroids[i][1]), int(centroids[i][0])

        # Check for overlaps
        if _should_exclude_spot(y, x, color_range, used_positions_global, 
                               used_positions_by_color[color_name], 
                               global_exclude_dist, same_color_dist):
            continue

        spot = SpotProperties(y, x, image, area)
        spots.append(spot)
        used_positions_global.add((y, x))
        used_positions_by_color[color_name].add((y, x))

    return spots

def _apply_special_processing(hsv, mask, color_name):
    """Apply special processing for certain colors."""
    if color_name == 'light_blue':
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        refine = ((s > 50) & (v > 150)).astype(np.uint8) * 255
        return cv2.bitwise_and(mask, refine)
    elif color_name == 'dark_gray':
        v = hsv[:, :, 2]
        refine = ((v > 60) & (v < 180)).astype(np.uint8) * 255
        return cv2.bitwise_and(mask, refine)
    return mask

def _should_exclude_spot(y, x, color_range, used_global, used_same_color, 
                        global_dist, same_color_dist):
    """Check if spot should be excluded due to overlaps."""
    # Check global overlap exclusion
    if color_range.get('exclude_overlap'):
        for uy, ux in used_global:
            if abs(y - uy) < global_dist and abs(x - ux) < global_dist:
                return True

    # Check same-color overlap
    for cy, cx in used_same_color:
        if abs(y - cy) < same_color_dist and abs(x - cx) < same_color_dist:
            return True

    return False

def _unify_close_spots(spots, image, color_name=''):
    """Merge nearby spots using DBSCAN clustering."""
    if not spots:
        return spots
    
    eps = DBSCAN_EPS.get(color_name, DBSCAN_EPS['default'])
    coords = np.array([[sp.centroid[1], sp.centroid[0]] for sp in spots], dtype=np.float32)
    
    clusterer = DBSCAN(eps=eps, min_samples=1)
    labels = clusterer.fit_predict(coords)

    merged_spots = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        group = [spots[i] for i in indices]
        
        # Calculate average position and total area
        mean_x = np.mean([sp.centroid[1] for sp in group])
        mean_y = np.mean([sp.centroid[0] for sp in group])
        total_area = sum(sp.area for sp in group)
        
        merged_spot = SpotProperties(int(round(mean_y)), int(round(mean_x)), image, total_area)
        merged_spots.append(merged_spot)
    
    return merged_spots

def apply_color_fallbacks(spots_dict, key_colors):
    """Apply color fallback mapping for colors not in key region."""
    filtered_spots = {}
    
    for color_name, spots in spots_dict.items():
        if key_colors and color_name not in key_colors:
            if color_name in COLOR_FALLBACKS and COLOR_FALLBACKS[color_name] in key_colors:
                fallback = COLOR_FALLBACKS[color_name]
                print(f"Remapping {color_name} => {fallback}")
                filtered_spots.setdefault(fallback, []).extend(spots)
            else:
                print(f"Skipping {color_name}; no valid fallback")
        else:
            filtered_spots.setdefault(color_name, []).extend(spots)
    
    return filtered_spots

def cluster_spots_by_color(spots_by_color, k_clusters):
    """Cluster all spots into K groups based on color similarity."""
    all_spots = []
    for spots in spots_by_color.values():
        all_spots.extend(spots)
    
    if not all_spots or k_clusters < 1:
        return spots_by_color
    
    # Extract color features (LAB color space)
    color_features = []
    for spot in all_spots:
        patch_bgr = cv2.cvtColor(spot.intensity_image, cv2.COLOR_RGB2BGR)
        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        lab_mean = [np.mean(patch_lab[:, :, i]) for i in range(3)]
        color_features.append(lab_mean)

    color_features = np.array(color_features, dtype=np.float32)
    n_spots = len(all_spots)
    
    if k_clusters > n_spots:
        k_clusters = n_spots
    if k_clusters == 1:
        return {"cluster_1": all_spots}

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(color_features)

    # Group spots by cluster
    clustered = {}
    for i, spot in enumerate(all_spots):
        cluster_name = f"cluster_{labels[i] + 1}"
        clustered.setdefault(cluster_name, []).append(spot)
    
    return clustered

def calculate_spot_statistics(spots_by_color, injection_site):
    """Calculate spatial statistics for all detected spots."""
    if not spots_by_color:
        return {}

    # Collect all spots with their colors
    all_spots = []
    for color, spots in spots_by_color.items():
        for spot in spots:
            all_spots.append((color, spot))

    if not all_spots:
        return {}

    # Extract coordinates and colors
    coords = np.array([(spot.centroid[1], spot.centroid[0]) for _, spot in all_spots])
    colors = [color for color, _ in all_spots]

    results_by_color = {}
    for color in spots_by_color:
        results_by_color[color] = []

    for color, spots in spots_by_color.items():
        for i, spot in enumerate(spots):
            cx, cy = spot.centroid[1], spot.centroid[0]
            
            # Calculate distances
            dist_to_injection = distance.euclidean((cx, cy), injection_site)
            
            # Find nearest neighbor (any color)
            if len(coords) > 1:
                distances = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
                spot_index = next(j for j, (c, s) in enumerate(all_spots) if s is spot)
                distances[spot_index] = np.inf
                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]
                nearest_color = colors[nearest_idx]
            else:
                nearest_dist = np.nan
                nearest_color = ""

            # Find nearest same-color neighbor
            same_color_coords = [coords[j] for j, c in enumerate(colors) if c == color]
            if len(same_color_coords) > 1:
                same_color_coords = np.array(same_color_coords)
                distances = np.sqrt((same_color_coords[:, 0] - cx)**2 + (same_color_coords[:, 1] - cy)**2)
                distances[distances < 1e-7] = np.inf  # Exclude self
                same_color_dist = np.min(distances) if not np.all(np.isinf(distances)) else np.nan
            else:
                same_color_dist = np.nan

            # Calculate intensity
            intensity = np.mean(spot.intensity_image)

            # Store results
            result = {
                'Spot_ID': i + 1,
                'Centroid_X': cx,
                'Centroid_Y': cy,
                'Area': spot.area,
                'Distance_to_Injection': dist_to_injection,
                'Nearest_Neighbor_Distance': nearest_dist,
                'Nearest_Neighbor_Color': nearest_color,
                'Nearest_Same_Color_Distance': same_color_dist,
                'Intensity': intensity
            }
            results_by_color[color].append(result)

    # Convert to DataFrames
    final_results = {}
    for color, data in results_by_color.items():
        final_results[color] = pd.DataFrame(data)
    
    return final_results
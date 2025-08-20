
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

"""Visualization functions for creating plots and images."""

import os
import traceback
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

from config import BGR_COLORS, parse_base_color

def create_spot_plot(image, df, color_name, injection_site, output_path, show_arrows=False):
    """Create visualization of spots with optional arrows."""
    try:
        bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        
        # Draw injection site marker
        inj_x, inj_y = int(injection_site[0]), int(injection_site[1])
        cv2.drawMarker(bgr, (inj_x, inj_y), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(bgr, (inj_x, inj_y), 5, (255, 255, 255), -1)

        # Get color for spots
        base_color = parse_base_color(color_name)
        spot_color = BGR_COLORS.get(base_color, (255, 255, 255))

        # Draw spots in batches to avoid memory issues
        batch_size = 25
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                cx, cy = int(row['Centroid_X']), int(row['Centroid_Y'])
                
                # Draw spot
                cv2.circle(bgr, (cx, cy), 5, spot_color, -1)
                
                # Draw spot ID
                cv2.putText(bgr, str(row['Spot_ID']), (cx + 5, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw arrow if requested
                if show_arrows:
                    cv2.arrowedLine(bgr, (inj_x, inj_y), (cx, cy), 
                                   spot_color, 2, tipLength=0.05)

        # Add title
        title = f"{color_name} ({len(df)} spots)"
        cv2.putText(bgr, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, spot_color, 2)
        
        # Save image
        cv2.imwrite(output_path, bgr)
        return True
        
    except Exception as e:
        print(f"Error creating spot plot: {e}")
        return False

def create_overview_plot(image, results_by_color, injection_site, output_path):
    """Create overview plot with all colors."""
    try:
        bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        
        # Draw injection site
        inj_x, inj_y = int(injection_site[0]), int(injection_site[1])
        cv2.drawMarker(bgr, (inj_x, inj_y), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(bgr, (inj_x, inj_y), 5, (255, 255, 255), -1)

        # Draw all spots with arrows
        for color_name, df in results_by_color.items():
            if df.empty:
                continue
                
            base_color = parse_base_color(color_name)
            spot_color = BGR_COLORS.get(base_color, (255, 255, 255))

            for _, row in df.iterrows():
                cx, cy = int(row['Centroid_X']), int(row['Centroid_Y'])
                cv2.circle(bgr, (cx, cy), 5, spot_color, -1)
                cv2.arrowedLine(bgr, (inj_x, inj_y), (cx, cy), spot_color, 2, tipLength=0.05)

        cv2.imwrite(output_path, bgr)
        print(f"Created overview plot: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating overview plot: {e}")
        return False

def create_all_visualizations(image, results_by_color, injection_site, output_dir, base_filename):
    """Create all spot visualizations."""
    created_files = []
    
    for color_name, df in results_by_color.items():
        if df.empty:
            continue
            
        # Create plot without arrows
        no_arrows_path = os.path.join(output_dir, f"{base_filename}_{color_name.lower()}_spots.png")
        if create_spot_plot(image, df, color_name, injection_site, no_arrows_path, show_arrows=False):
            created_files.append(no_arrows_path)
            print(f"Created: {no_arrows_path}")

        # Create plot with arrows
        arrows_path = os.path.join(output_dir, f"{base_filename}_{color_name.lower()}_arrows.png")
        if create_spot_plot(image, df, color_name, injection_site, arrows_path, show_arrows=True):
            created_files.append(arrows_path)
            print(f"Created: {arrows_path}")

    # Create overview plot
    overview_path = os.path.join(output_dir, f"{base_filename}_overview_arrows.png")
    if create_overview_plot(image, results_by_color, injection_site, overview_path):
        created_files.append(overview_path)

    return created_files

def create_statistical_plots(results, output_dir, base_filename):
    """Create statistical violin plots."""
    print("Creating statistical plots...")
    
    try:
        # Combine all data
        all_data = []
        for color_name, df in results.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['Color'] = color_name
                all_data.append(df_copy)

        if not all_data:
            print("No data for statistical plots")
            return []

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")

        os.makedirs(output_dir, exist_ok=True)
        plot_paths = []

        # Generate color palette
        color_names = sorted(combined_df['Color'].unique())
        palette = _generate_color_palette(color_names)

        # Create plots
        plots_config = [
            ('Distance_to_Injection', 'Distances to Injection Site', 'distance_distribution'),
            ('Nearest_Neighbor_Distance', 'Nearest Neighbor Distance (All Spots)', 'neighbor_distribution'),
            ('Nearest_Same_Color_Distance', 'Nearest Same-Color Distance', 'same_color_distribution')
        ]

        for column, title, filename in plots_config:
            plot_path = _create_violin_plot(
                combined_df, column, title, color_names, palette, 
                output_dir, f"{base_filename}_{filename}.svg"
            )
            if plot_path:
                plot_paths.append(plot_path)

        return plot_paths

    except Exception as e:
        print(f"Error creating statistical plots: {e}")
        traceback.print_exc()
        return []
    finally:
        plt.close('all')

def _generate_color_palette(color_names):
    """Generate consistent color palette for plots."""
    palette = []
    for name in color_names:
        seed = abs(hash(name)) % 9999
        np.random.seed(seed)
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        palette.append(hex_color)
    return palette

def _create_violin_plot(data, y_column, title, color_names, palette, output_dir, filename):
    """Create a single violin plot with significance bars."""
    try:
        plt.figure(figsize=(15, 8))
        
        # Calculate significance positions
        sig_positions = _calculate_significance_positions(data, y_column, color_names)
        
        # Create violin plot
        ax = sns.violinplot(
            data=data, x='Color', y=y_column, order=color_names, palette=palette,
            cut=0, bw=0.5, gridsize=50, inner='box'
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.title(title, pad=20)
        plt.xlabel('Color')
        plt.ylabel('Distance (pixels)')
        
        # Add significance bars
        _add_significance_bars(ax, sig_positions)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='svg')
        plt.close()
        
        print(f"Created plot: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"Error creating {filename}: {e}")
        plt.close()
        return None

def _calculate_significance_positions(data, y_column, color_names):
    """Calculate positions for significance bars."""
    if data[y_column].empty:
        return []
    
    y_max = data[y_column].max()
    bar_height = y_max * 0.05 if y_max > 0 else 1
    pairs = list(combinations(color_names, 2))
    sig_positions = []
    
    for i, (color1, color2) in enumerate(pairs):
        data1 = data[data['Color'] == color1][y_column]
        data2 = data[data['Color'] == color2][y_column]
        
        if len(data1) > 0 and len(data2) > 0:
            try:
                _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                if p_value < 0.05:
                    x1 = color_names.index(color1)
                    x2 = color_names.index(color2)
                    y = y_max + (bar_height * (i + 1))
                    
                    if p_value < 0.001:
                        symbol = '***'
                    elif p_value < 0.01:
                        symbol = '**'
                    else:
                        symbol = '*'
                    
                    sig_positions.append((x1, x2, y, symbol))
            except:
                pass
    
    return sig_positions

def _add_significance_bars(ax, sig_positions):
    """Add significance bars to plot."""
    for x1, x2, y, symbol in sig_positions:
        ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
        ax.text((x1 + x2) / 2, y, symbol, ha='center', va='bottom')

def save_results_to_excel(results, output_dir, base_filename):
    """Save analysis results to Excel file."""
    excel_path = os.path.join(output_dir, f"{base_filename}_analysis_results.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        for color_name, df in results.items():
            if len(df) > 0:
                df.to_excel(writer, sheet_name=color_name, index=False)
                print(f"Saved {len(df)} {color_name} spots to Excel")
    
    print(f"Results saved to: {excel_path}")
    return excel_path

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

"""GUI components for the spot analysis application."""

import os
import sys
import traceback
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QWidget, QProgressBar, QMessageBox, QLineEdit, 
    QTextEdit, QFrame, QRubberBand
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QSize

from image_processing import load_image, find_largest_square, crop_image, extract_foreground, adjust_injection_site_for_crop
from spot_detection import detect_key_colors, detect_color_spots, apply_color_fallbacks, cluster_spots_by_color, calculate_spot_statistics
from visualization import create_all_visualizations, create_statistical_plots, save_results_to_excel

class ROISelector(QLabel):
    """Widget for selecting injection site region of interest."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        self.selection_complete = False
        self.selected_rect = None
        self.setMouseTracking(True)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid darkgray;
                border-radius: 5px;
            }
        """)

    def mousePressEvent(self, event):
        self.origin = event.pos()
        self.current_rect = QRect(self.origin, QSize(0, 0))
        self.rubberband.setGeometry(self.current_rect)
        self.rubberband.show()

    def mouseMoveEvent(self, event):
        if self.origin:
            self.current_rect = QRect(self.origin, event.pos()).normalized()
            self.rubberband.setGeometry(self.current_rect)

    def mouseReleaseEvent(self, event):
        if self.origin:
            final_rect = QRect(self.origin, event.pos()).normalized()
            if final_rect.width() > 10 and final_rect.height() > 10:
                self.selected_rect = final_rect
                self.selection_complete = True
                self.rubberband.setGeometry(final_rect)
            self.origin = None

    def reset_selection(self):
        self.rubberband.hide()
        self.origin = None
        self.selection_complete = False
        self.selected_rect = None

    def set_images(self, image_paths):
        """Display first available image or overview if available."""
        if not image_paths:
            self.clear()
            self.setText("No Image")
            return
            
        # Prefer overview image if available
        overview_image = None
        for path in image_paths:
            if 'overview' in path.lower():
                overview_image = path
                break
        
        chosen_image = overview_image if overview_image else image_paths[0]
        pixmap = QPixmap(chosen_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class ProgressStream:
    """Stream for capturing console output and emitting progress signals."""
    
    def __init__(self, original_stream, progress_signal, output_signal):
        self.original_stream = original_stream
        self.progress_signal = progress_signal
        self.output_signal = output_signal

    def write(self, message):
        if not message:
            return
        self.original_stream.write(message)
        self.output_signal.emit(message.strip())
        
        # Simple progress tracking based on keywords
        progress_keywords = {
            'reading image': 10,
            'extracting foreground': 20,
            'detecting spots': 30,
            'detecting which colors': 40,
            'merging fallback': 50,
            'clustering': 60,
            'calculating statistics': 70,
            'creating': 80,
            'saving results': 90,
            'generating statistical': 95
        }
        
        message_lower = message.lower()
        for keyword, progress in progress_keywords.items():
            if keyword in message_lower:
                self.progress_signal.emit(progress)
                break

    def flush(self):
        self.original_stream.flush()

class AnalysisThread(QThread):
    """Thread for running analysis without blocking the GUI."""
    
    progress_update = pyqtSignal(int)
    console_output = pyqtSignal(str)
    analysis_complete = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, output_dir, injection_site, 
                 fraction=0.25, buffer=50, debug=False, k_clusters=0):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.injection_site = injection_site
        self.fraction = fraction
        self.buffer = buffer
        self.debug = debug
        self.k_clusters = k_clusters
        
        self.results = {}
        self.subfolder_path = ""
        self.base_filename = ""

    def run(self):
        original_stdout = sys.stdout
        try:
            # Redirect stdout for progress tracking
            sys.stdout = ProgressStream(original_stdout, self.progress_update, self.console_output)

            print("Reading image...")
            image = load_image(self.image_path)

            print("Finding largest bounding square...")
            bbox = find_largest_square(image)
            if bbox is None:
                print("No bounding square found, using original image")
                cropped_image = image
            else:
                print(f"Found bounding square: {bbox}")
                cropped_image = crop_image(image, bbox)

            print("Extracting foreground...")
            foreground = extract_foreground(cropped_image)

            print("Detecting which colors are in the key region...")
            key_colors = detect_key_colors(foreground, self.fraction, self.buffer)
            print(f"Key region colors: {key_colors}")

            print("Detecting spots in the main region...")
            spots_dict = detect_color_spots(foreground, self.fraction, self.buffer)
            
            if not spots_dict:
                print("No spots found in main region")
                self.analysis_complete.emit(True)
                return

            print("Merging fallback colors...")
            filtered_spots = apply_color_fallbacks(spots_dict, key_colors)
            
            if not filtered_spots:
                print("No valid spots after applying fallbacks")
                self.analysis_complete.emit(True)
                return

            # Apply clustering if requested
            if self.k_clusters > 1:
                print(f"Clustering {sum(len(spots) for spots in filtered_spots.values())} spots into {self.k_clusters} clusters")
                final_spots = cluster_spots_by_color(filtered_spots, self.k_clusters)
            else:
                final_spots = filtered_spots

            print("Calculating statistics...")
            # Adjust injection site for cropped image
            adjusted_injection = adjust_injection_site_for_crop(self.injection_site, bbox)
            if bbox is not None:
                h, w = foreground.shape[:2]
                adjusted_injection = (
                    max(0, min(w - 1, adjusted_injection[0])),
                    max(0, min(h - 1, adjusted_injection[1]))
                )

            self.results = calculate_spot_statistics(final_spots, adjusted_injection)
            
            # Set up output directory
            self.base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
            self.subfolder_path = os.path.join(self.output_dir, self.base_filename)
            os.makedirs(self.subfolder_path, exist_ok=True)

            print("Creating visualizations...")
            create_all_visualizations(
                cropped_image, self.results, adjusted_injection, 
                self.subfolder_path, self.base_filename
            )

            print("Saving results to Excel...")
            save_results_to_excel(self.results, self.subfolder_path, self.base_filename)

            self.progress_update.emit(100)
            self.analysis_complete.emit(True)

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

        finally:
            sys.stdout = original_stdout

class SpotAnalysisGUI(QMainWindow):
    """Main GUI window for the spot analysis application."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.reset_parameters()
        self.analysis_thread = None

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Spot Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # File selection section
        file_layout = QHBoxLayout()
        
        # Input image selection
        file_layout.addWidget(QLabel("Input Image:"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        file_layout.addWidget(self.input_path_edit)
        
        browse_input_btn = QPushButton("Browse Image")
        browse_input_btn.clicked.connect(self.browse_input_image)
        file_layout.addWidget(browse_input_btn)

        # Output directory selection
        file_layout.addWidget(QLabel("Output Directory:"))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        file_layout.addWidget(self.output_path_edit)
        
        browse_output_btn = QPushButton("Browse Output")
        browse_output_btn.clicked.connect(self.browse_output_directory)
        file_layout.addWidget(browse_output_btn)

        main_layout.addLayout(file_layout)

        # Parameters section
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Key Fraction:"))
        self.fraction_edit = QLineEdit("0.25")
        params_layout.addWidget(self.fraction_edit)
        
        params_layout.addWidget(QLabel("Buffer:"))
        self.buffer_edit = QLineEdit("50")
        params_layout.addWidget(self.buffer_edit)
        
        params_layout.addWidget(QLabel("K-Means Clusters (0=none):"))
        self.kmeans_edit = QLineEdit("0")
        params_layout.addWidget(self.kmeans_edit)
        
        self.debug_check = QPushButton("Debug Mode", checkable=True)
        params_layout.addWidget(self.debug_check)

        main_layout.addLayout(params_layout)

        # Image display section
        images_layout = QHBoxLayout()
        
        self.original_image_display = ROISelector()
        self.original_image_display.setMinimumSize(500, 500)
        images_layout.addWidget(self.original_image_display)
        
        self.result_image_display = ROISelector()
        self.result_image_display.setMinimumSize(500, 500)
        images_layout.addWidget(self.result_image_display)

        main_layout.addLayout(images_layout)

        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(150)
        main_layout.addWidget(self.console_output)

        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.select_injection_btn = QPushButton("Select Injection Site")
        self.select_injection_btn.clicked.connect(self.select_injection_site)
        controls_layout.addWidget(self.select_injection_btn)

        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)

        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.run_analysis_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_all)
        controls_layout.addWidget(self.reset_btn)

        main_layout.addLayout(controls_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.input_image_path = ""
        self.output_directory = ""
        self.injection_site = None

    def browse_input_image(self):
        """Browse and select input image file."""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Select Input Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        
        if image_path:
            self.input_image_path = image_path
            self.input_path_edit.setText(image_path)
            
            # Display image
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.original_image_display.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image_display.setPixmap(scaled_pixmap)
            self.original_image_display.reset_selection()

    def browse_output_directory(self):
        """Browse and select output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_directory = directory
            self.output_path_edit.setText(directory)

    def select_injection_site(self):
        """Select injection site from image."""
        if not self.input_image_path:
            QMessageBox.warning(self, "Error", "Please select an input image first.")
            return
            
        if not self.original_image_display.selection_complete:
            QMessageBox.warning(self, "Error", "Please draw a rectangle around the injection site.")
            return

        selected_rect = self.original_image_display.selected_rect
        reply = QMessageBox.question(
            self, 'Confirm Injection Site',
            'Use this selection as the injection site?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            self.original_image_display.reset_selection()
            return

        # Convert selection to image coordinates
        original_pixmap = QPixmap(self.input_image_path)
        scaled_pixmap = original_pixmap.scaled(
            self.original_image_display.size(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        scale_x = original_pixmap.width() / scaled_pixmap.width()
        scale_y = original_pixmap.height() / scaled_pixmap.height()

        # Calculate center of selection
        center_x = selected_rect.x() + selected_rect.width() / 2
        center_y = selected_rect.y() + selected_rect.height() / 2
        
        # Convert to image coordinates
        image_x = int(center_x * scale_x)
        image_y = int(center_y * scale_y)

        # Validate coordinates
        temp_image = load_image(self.input_image_path)
        h, w = temp_image.shape[:2]
        image_x = max(0, min(w - 1, image_x))
        image_y = max(0, min(h - 1, image_y))

        self.injection_site = (image_x, image_y)
        QMessageBox.information(self, "Injection Site Selected", 
                               f"Injection site set to: {self.injection_site}")

    def run_analysis(self):
        """Run the spot analysis."""
        # Validate inputs
        if not self.input_image_path:
            QMessageBox.warning(self, "Error", "Please select an input image.")
            return
        if not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return
        if not self.injection_site:
            QMessageBox.warning(self, "Error", "Please select an injection site.")
            return

        # Get parameters
        try:
            fraction = float(self.fraction_edit.text())
            buffer = int(self.buffer_edit.text())
            k_clusters = int(self.kmeans_edit.text())
            debug = self.debug_check.isChecked()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please check parameter values.")
            return

        # Disable controls during analysis
        self.run_analysis_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.console_output.clear()

        # Start analysis thread
        self.analysis_thread = AnalysisThread(
            self.input_image_path, self.output_directory, self.injection_site,
            fraction, buffer, debug, k_clusters
        )
        
        self.analysis_thread.progress_update.connect(self.update_progress)
        self.analysis_thread.console_output.connect(self.update_console)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)

        self.analysis_thread.start()

    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def update_console(self, message):
        """Update console output."""
        self.console_output.append(message)
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_analysis_complete(self, success):
        """Handle analysis completion."""
        self.run_analysis_btn.setEnabled(True)
        
        if success:
            # Display result images
            base_filename = self.analysis_thread.base_filename
            subfolder = self.analysis_thread.subfolder_path
            results = self.analysis_thread.results

            # Find output images
            output_images = []
            if os.path.exists(subfolder):
                for file in os.listdir(subfolder):
                    if file.endswith('.png') and base_filename in file:
                        output_images.append(os.path.join(subfolder, file))

            self.result_image_display.set_images(output_images)

            QMessageBox.information(self, "Success", "Analysis completed successfully!")
            
            # Generate statistical plots
            print("Generating statistical plots...")
            create_statistical_plots(results, subfolder, base_filename)

    def on_analysis_error(self, error_message):
        """Handle analysis error."""
        self.run_analysis_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_message)

    def reset_all(self):
        """Reset all inputs and displays."""
        self.reset_parameters()
        
        # Clear UI elements
        self.input_path_edit.clear()
        self.output_path_edit.clear()
        self.console_output.clear()
        self.progress_bar.setValue(0)
        
        # Reset parameter fields
        self.fraction_edit.setText("0.25")
        self.buffer_edit.setText("50")
        self.kmeans_edit.setText("0")
        self.debug_check.setChecked(False)
        
        # Clear image displays
        self.original_image_display.clear()
        self.original_image_display.reset_selection()
        self.result_image_display.clear()
        self.result_image_display.reset_selection()

        QMessageBox.information(self, "Reset", "All settings have been reset.")
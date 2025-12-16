from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
from czifile import CziFile
import xml.etree.ElementTree as ET
import cv2
from dataclasses import dataclass
import numpy as np
import statistics
from math import hypot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
from scipy.spatial import KDTree
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

debug = False
display = False 

settings = []
if debug:
    settings = "d1_kinetix"
else:
    settings = input("Enter chip used (options: d1_kinetix, bd_kinetix, c5_kinetix, c5_axio, c2_kinetix): ").strip().lower()

match settings.lower():
    case "d1_kinetix":
        print("loading preset for d1_kinetix")
        min_diameter = 60
        max_diameter = 63
        roi_inner = -30
        roi_outer = 6
        bckg_inner = 80 
        bckg_outer = 120
        moving_avg_n = 10
        p1 = 50
        p2 = 20
        dp = 1.5
        min_distance = max_diameter

    case "bd_kinetix":
        print("loading preset for bd_kinetix")
        min_diameter = 23
        max_diameter = 26
        roi_inner = -15 #5
        roi_outer = 4
        bckg_inner = 5  #3
        bckg_outer = 6  #6
        moving_avg_n = 10
        p1 = 50
        p2 = 30
        dp = 1
        min_distance = max_diameter

    case "c5_kinetix":
        print("loading preset for c3_kinetix")
        min_diameter = 16
        max_diameter = 20
        roi_inner = -15
        roi_outer = -5
        bckg_inner = 15
        bckg_outer = 30
        moving_avg_n = 10
        p1 = 50
        p2 = 20
        dp = 1.0
        min_distance = max_diameter

    case "c2_kinetix":
        print("loading preset for c2_kinetix")
        min_diameter = 20
        max_diameter = 30
        roi_inner = -20
        roi_outer = -7
        bckg_inner = 30
        bckg_outer = 50
        moving_avg_n = 10
        p1 = 50
        p2 = 20
        dp = 1.0

    case "c5_axio":
        print("loading preset for c5_axio")
        min_diameter = 16
        max_diameter = 20
        roi_inner = -15
        roi_outer = -5
        bckg_inner = 25
        bckg_outer = 45
        moving_avg_n = 2
        p1 = 50
        p2 = 20
        dp = 1.0

start = time.time()

class Circle:
    def __init__(self, x, y, r=5, parent=None, index=None):
        self.x = x
        self.y = y
        self.r = r
        self.parent = parent
        self.index = index
        self.fluorescence = {}

class FindFolders:
    """
    Unified class to locate microscopy data for downstream analysis.

    Behavior automatically adapts based on file_type:

    - TIF datasets:
        Each channel is typically saved as a separate .tif file inside
        a folder that represents one biological sample. Therefore:
            * The class returns a LIST OF FOLDERS.
            * Each folder contains multiple TIF images representing channels.

        Returned format:
            parent_dir: str
            folders: List[str]     # folders containing TIF channels

    - CZI datasets:
        A single .czi file typically contains all channels as internal layers.
        Therefore:
            * The class returns a LIST OF FILE PATHS.
            * Each file is a complete imaging dataset.

        Returned format:
            parent_dir: str
            folders: List[str]     # paths to .czi files
    """

    def __init__(self, file_type: str):
        """
        Initialize the finder.

        Parameters
        ----------
        file_type : str
            't' for TIFF datasets
            'c' for CZI datasets
        """
        # Tkinter setup
        self.root = Tk()
        self.root.withdraw()
        self.root.attributes('-topmost', True)

        # Interpret file type
        if file_type == "t":
            self.file_type = "tif"
            self.mode = "folders"
        elif file_type == "c":
            self.file_type = "czi"
            self.mode = "files"
        else:
            raise ValueError("file_type must be 't' (TIF) or 'c' (CZI).")

        self.folders = []

    def select_directory(self):
        """Show topmost directory picker and return the selected path."""
        return askdirectory(title="Select Parent Directory")

    def search(self, root_directory):
        """
        Search for datasets inside the root directory.

        For TIF:
            Collect folders that contain at least one TIF file.
        For CZI:
            Collect each CZI file directly.
        """
        self.folders.clear()

        if self.mode == "folders":  # TIF mode
            for dirpath, dirnames, filenames in os.walk(root_directory):
                if any(f.lower().endswith(".tif") for f in filenames):
                    self.folders.append(dirpath)

        elif self.mode == "files":  # CZI mode
            for dirpath, dirnames, filenames in os.walk(root_directory):
                for f in filenames:
                    if f.lower().endswith(".czi"):
                        self.folders.append(os.path.join(dirpath, f))

        return bool(self.folders)

    def run(self):
        """
        Main workflow.

        Returns
        -------
        tuple
            (parent_directory: str | None,
             folders: List[str])

        In TIF mode:
            folders = folder paths
        In CZI mode:
            folders = file paths
        """
        parent = self.select_directory()
        if not parent:
            print("No directory selected. Exiting.")
            return None, []

        self.search(parent)

        if not self.folders:
            print(f"No {self.file_type.upper()} datasets found.")

        if debug:
            print(f"Found {len(self.folders)} {self.file_type.upper()} datasets.")

        return parent, self.folders

class FileOrganization:
    """
    Organizes microscopy data depending on file_type:

    TIF mode ('t'):
        - Input: a folder containing multiple .tif files
        - Output:
            * list of TIFF files
            * brightfield file
            * fluorescent files
            * created channel folders

    CZI mode ('c'):
        - Input: a single .czi file
        - Output:
            * the CZI file path
            * sample name
            * (optionally) create an output folder for extracted channels
            * list of channels in the image data
    """

    def __init__(self, folder_or_file: str, file_type: str):
        self.path = folder_or_file
        self.czi_file = None

        # TIFF-specific
        self.tiff_files = []
        self.bf_image_file = None
        self.fl_image_files = []
        self.created_folders = {}
        self.bright_folder = None

        # Interpret file type
        if file_type == "t":
            self.file_type = "tif"
            self.mode = "tif_mode"
        elif file_type == "c":
            self.file_type = "czi"
            self.mode = "czi_mode"
        else:
            raise ValueError("file_type must be 't' or 'c'.")

    # ---------------------- TIF MODE LOGIC --------------------------

    def organize_tiff_files(self):
        """
        Organize a folder full of .tif images into channel subfolders:
        Bright, DAPI, Cy5, EGFP, DsRed
        """
        folder = self.path

        self.tiff_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
        ]


        if not self.tiff_files:
            print(f"No TIFF images found in {folder}")
            return None

        # Identify brightfield
        bf = [f for f in self.tiff_files if "bright" in f.lower()]
        self.bf_image_file = bf[0] if bf else None

        # Separate fluorescent images
        self.fl_image_files = [
            f for f in self.tiff_files if "bright" not in f.lower()
        ]

        # Create channel folders
        keywords = ["bright", "cy5", "dapi", "egfp", "dsred"]

        """for tif in self.tiff_files:
            for key in keywords:
                if key in tif.lower():
                    new_folder = os.path.join(folder, key.title())
                    os.makedirs(new_folder, exist_ok=True)
                    self.created_folders[key.title()] = new_folder
                    break"""

        self.bright_folder = self.created_folders.get("Bright", None)
        #output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_data")
        #os.makedirs(output_folder, exist_ok=True)

        return {
            "tiff_files": self.tiff_files, #returns full path
            "brightfield": self.bf_image_file, #returns full path
            "fluorescent": self.fl_image_files, #returns full path
            #"created_folders": self.created_folders, #returns full path
            "bright_folder": self.bright_folder, #returns full path
            "output_folder": folder
        }

    # ---------------------- CZI MODE LOGIC --------------------------

    def organize_czi_file(self):
        self.czi_file = self.path

        if not self.czi_file.lower().endswith(".czi"):
            print("Error: CZI mode expected a .czi file.")
            return None

        sample_name = os.path.splitext(os.path.basename(self.czi_file))[0]
        output_folder = os.path.join(os.path.dirname(self.czi_file), sample_name)
        os.makedirs(output_folder, exist_ok=True)

        with CziFile(self.czi_file) as czi:
            img = czi.asarray()
            axes = czi.axes  # CYX0

            if debug:
                print("Data shape:", img.shape)
                print("Axes:", axes)

            # Parse XML metadata
            md_xml = czi.metadata()
            root = ET.fromstring(md_xml)

            # Extract channel names
            channel_names = []
            for ch in root.findall(".//Information/Image/Dimensions/Channels/Channel"):
                name = ch.get("Name") or ch.get("ID") or "Unnamed"
                channel_names.append(name)

            n_img_channels = img.shape[0]
            actual_channels = channel_names[:n_img_channels]

            if debug:
                print("Channel names:", channel_names)
                print("Actual channels in image data:", actual_channels)

            # Remove useless trailing dimension: (C, Y, X, 1) → (C, Y, X)
            img = img[:, :, :, 0]

            # Build mapping: channel name → array
            channel_images = {
                name: img[i, :, :]
                for i, name in enumerate(actual_channels)
            }

        return {
            "czi_file": self.czi_file,
            "sample_name": sample_name,
            "output_folder": output_folder,
            "channels": actual_channels,
            "images": channel_images,
        }



    # ---------------------- CONTROLLER --------------------------

    def run(self):
        """Run the correct mode depending on file type."""
        if self.mode == "tif_mode":
            return self.organize_tiff_files()

        elif self.mode == "czi_mode":
            return self.organize_czi_file()

class UniversalImageLoader:
    """Load TIFF or CZI images into NumPy arrays."""

    def __init__(self, file_type: str):
        self.file_type = file_type.lower()

    def load(self, path: str, channel_index: int = 0):
        if self.file_type == "t":
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)

        elif self.file_type == "c":
            return self.load_czi_channel(path, channel_index)

        else:
            raise ValueError("file_type must be 't' or 'c'")

    def load_czi_channel(self, czi_path, channel_index=0):
        with CziFile(czi_path) as czi:
            img = czi.asarray()
            axes = czi.axes  # Should be CYX0

            # --- YOUR CASE: ALWAYS "CYX0" ---
            if axes == "CYX0":
                channel = img[channel_index, :, :, 0]
                return np.squeeze(channel)

            raise ValueError(f"Unexpected CZI axis order: {axes}")

    def load_fluorescent_images(self, fl_image_files=None, czi_file=None, channel_map=None):
        images = {key: None for key in ['Cy5', 'DAPI', 'DsRed', 'EGFP']}

        if self.file_type == "t":
            if fl_image_files is None:
                raise ValueError("fl_image_files must be provided for TIFFs")

            for fl_file in fl_image_files:
                for key in images.keys():
                    if key.lower() in fl_file.lower():
                        images[key] = cv2.imread(fl_file, cv2.IMREAD_UNCHANGED)

        elif self.file_type == "c":
            if czi_file is None or channel_map is None:
                raise ValueError("czi_file and channel_map must be provided for CZI")

            for key in images.keys():
                idx = channel_map.get(key)
                if idx is not None:
                    images[key] = self.load(czi_file, channel_index=idx)

        return images

class CircleDetection:
    def __init__(self, bf_image):
        self.minrad = min_diameter
        self.maxrad = max_diameter
        self.roi_inner = roi_inner
        self.roi_outer = roi_outer
        self.bckg_inner = bckg_inner
        self.bckg_outer = bckg_outer
        self.p1 = p1
        self.p2 = p2
        self.dp = dp
        self.mindist = min_distance
        
        self.bf_image = bf_image

    def detect_circles(self):
        """Detect outer circles and return them as Circle objects."""
        bf_image = cv2.medianBlur(self.bf_image, 5)

        circles = cv2.HoughCircles(
            bf_image,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.mindist,
            param1=self.p1,
            param2=self.p2,
            minRadius=self.minrad,
            maxRadius=self.maxrad
        )

        outer_circles_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0]:
                outer_circles_list.append(Circle(x=int(x), y=int(y), r=int(r)))
        
        return outer_circles_list

    def detect_inner_circles(self, outer_circles):
        """Detect inner circles for each outer circle and store parent references."""
        if not outer_circles:
            return []

        inner_circles_list = []

        for outer in outer_circles:
            x, y, r = outer.x, outer.y, outer.r
            x1, x2 = x - r, x + r
            y1, y2 = y - r, y + r

            roi = self.bf_image[y1:y2, x1:x2]

            # Mask the ROI
            mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.circle(mask, (r, r), r, 255, -1)
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

            inner_circles = cv2.HoughCircles(
                roi_masked,
                cv2.HOUGH_GRADIENT,
                dp=self.dp,
                minDist=self.mindist,
                param1=self.p1 - 10,
                param2=self.p2,
                minRadius=self.minrad - 23,
                maxRadius=self.maxrad - 22
            )

            if inner_circles is None:
                if debug:
                    print(f"No inner circle detected for outer circle at ({x}, {y}) with radius {r}")
                continue

            inner_circles = np.uint16(np.around(inner_circles))
            for cx, cy, cr in inner_circles[0]:
                full_x = int(cx) + x1
                full_y = int(cy) + y1
                circle_obj = Circle(
                    x=full_x,
                    y=full_y,
                    r=int(cr),
                    parent=outer  # reference the outer circle
                )
                inner_circles_list.append(circle_obj)

        return inner_circles_list

    def run(self):
        """Return outer circles alone if 'bd' in settings, otherwise outer + inner circles."""
        outer_circles = self.detect_circles()

        if "bd" in settings.lower():
            return outer_circles
        else:
            inner_circles = self.detect_inner_circles(outer_circles)
            return outer_circles, inner_circles

def display_circles(image, outer_circles, inner_circles, title= "Detected Circles"):
    scale = 0.4
    h, w = image.shape[:2]
    resized = cv2.resize(image.copy(), (int(w*scale), int(h*scale)))

    if len(resized.shape) == 2:
        disp = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else: 
        disp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    for c in outer_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (0,255,0), 1)
    for c in inner_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (0,0,255), 1)

    # Create legend handles manually
    green_patch = patches.Patch(color='green', label='Outer Circles')
    blue_patch   = patches.Patch(color='blue', label='Inner Circles')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(disp)
    plt.title(title)
    plt.axis("off")
    plt.legend(handles = [green_patch, blue_patch], loc='upper right')
    plt.show()

def disp_missing_circles(image, outer_circles, inner_circles, missing_circles, title= "Missing Circles"):
    scale = 0.4
    h, w = image.shape[:2]
    resized = cv2.resize(image.copy(), (int(w*scale), int(h*scale)))

    if len(resized.shape) == 2:
        disp = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else: 
        disp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    for c in outer_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (0,255,0), 1)
    for c in inner_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (0,0,255), 1)
    for c in missing_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (255,0,0), 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(disp)
    plt.title(title)
    plt.axis("off")
    plt.show()

def disp_removed_circles(image, removed_circles, title= "Removed Circles"):
    scale = 0.4
    h, w = image.shape[:2]
    resized = cv2.resize(image.copy(), (int(w*scale), int(h*scale)))

    if len(resized.shape) == 2:
        disp = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else: 
        disp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    for c in removed_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (255,0,0), 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(disp)
    plt.title(title)
    plt.axis("off")
    plt.show()

def disp_all_circles(image, inner_circles, title= "Final Circles"):
    scale = 0.4
    h, w = image.shape[:2]
    resized = cv2.resize(image.copy(), (int(w*scale), int(h*scale)))

    if len(resized.shape) == 2:
        disp = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else: 
        disp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Draw found circles in green
    for c in inner_circles:
        cv2.circle(disp, (int(c.x*scale), int(c.y*scale)), int(c.r*scale), (255, 0, 0), 1)
        if hasattr(c, "index") and c.index is not None:
            cv2.putText(disp, str(c.index), (int(c.x*scale), int(c.y*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            
    plt.figure(figsize=(8, 8))
    plt.imshow(disp)
    plt.title(title)
    plt.axis("off")
    plt.show()

def filter_circles_inside_image(circles, img_w, img_h):
    """
    Removes circles whose radius would extend outside image boundaries.
    Works with Circle objects having attributes x, y, r.
    """
    filtered = []
    removed = []

    for c in circles:
        if (c.x - c.r >= 0 and
            c.x + c.r < img_w and
            c.y - c.r >= 0 and
            c.y + c.r < img_h):
            filtered.append(c)
        else:
            removed.append(c)

    return filtered, removed

def sort_circles(circles: list, y_tolerance = max_diameter):
    """
    Sort circles into rows (top-to-bottom) and left-to-right within each row.
    Assigns 1-based .index to each Circle.
    Returns:
        rows: list of lists (rows of circles)
        sorted_circles: flat list of circles with indices
    """
    if not circles:
        return [], []

    circles_list = sorted(circles, key=lambda c: c.y)

    rows = []
    current_row = [circles_list[0]]

    for circ in circles_list[1:]:
        if abs(circ.y - current_row[0].y) <= y_tolerance:
            current_row.append(circ)
        else:
            rows.append(sorted(current_row, key=lambda c: c.x))
            current_row = [circ]

    rows.append(sorted(current_row, key=lambda c: c.x))

    # Flatten and assign indices
    sorted_circles = []
    for idx, circle in enumerate([c for row in rows for c in row], start=1):
        circle.index = idx
        sorted_circles.append(circle)

    return rows, sorted_circles

def find_missing_circles(rows, match_tolerance):
    """
    Detect missing circles in a mostly rectangular grid robustly.
    
    Args:
        rows: list of lists of Circle objects (already roughly sorted)
        match_tolerance: maximum distance to consider an expected circle matched
    
    Returns:
        List of (x, y) tuples for missing circles
    """

    if not rows or not any(rows):
        return []

    # --- Sort rows top to bottom, each row left to right ---
    rows = sorted(rows, key=lambda r: statistics.mean([c.y for c in r]))
    for r in rows:
        r.sort(key=lambda c: c.x)

    # Determine maximum number of columns
    max_cols = max(len(r) for r in rows)

    # --- Build column-wise median X positions ---
    column_xs = []
    for col_idx in range(max_cols):
        col_items = [r[col_idx].x for r in rows if len(r) > col_idx]
        if col_items:
            median_x = statistics.median(col_items)
            column_xs.append(median_x)
        else:
            column_xs.append(None)  # column missing in some rows

    missing = []

    # --- Check each row for missing circles ---
    for row in rows:
        row_y = statistics.mean([c.y for c in row])  # use mean Y for row
        for col_idx, median_x in enumerate(column_xs):
            if median_x is None:
                continue  # skip completely empty columns
            # Check if a detected circle exists close to this column X
            if not any(abs(c.x - median_x) <= match_tolerance for c in row):
                missing.append((median_x, row_y))

    return missing

class ROIcreation:
    def __init__(self, circles, fl_images, channel_names, bf_image):
        """
        circles: list of Circle objects
        fl_images: list of 2D NumPy arrays (same order as channel_names)
        channel_names: list of names (same order as fl_images)
        bf_image: grayscale or RGB brightfield image
        """
        self.circles = circles
        self.fluorescent_images = fl_images
        self.channel_names = channel_names
        self.bf_image = bf_image.copy()

    # ------------------------------------------
    # Mask creation (safe for float positions)
    # ------------------------------------------
    def compute_masks(self, circle):
        """
        Precomputes ROI and background masks ONCE per circle.
        Converts positions and radii to int for OpenCV.
        """
        h, w = self.fluorescent_images[0].shape[:2]

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        bkg_mask = np.zeros((h, w), dtype=np.uint8)

        # Radii (assumes roi_inner, roi_outer, bckg_inner, bckg_outer are defined globally)
        roi_in  = int(circle.r + roi_inner)
        roi_out = int(circle.r + roi_outer)
        bkg_in  = int(circle.r + bckg_inner)
        bkg_out = int(circle.r + bckg_outer)

        # Positions
        cx, cy = int(circle.x), int(circle.y)

        # Draw ROI ring
        cv2.circle(roi_mask, (cx, cy), roi_out, 1, -1)
        cv2.circle(roi_mask, (cx, cy), roi_in, 0, -1)

        # Draw Background ring
        cv2.circle(bkg_mask, (cx, cy), bkg_out, 1, -1)
        cv2.circle(bkg_mask, (cx, cy), bkg_in, 0, -1)

        return roi_mask, bkg_mask
    
    def extract_fluorescence_local_threshold(self, circle):
        """
        New method: local noise threshold per electrode.
        - Background pixels log-transformed
        - Threshold = mean + 3*SD
        - Subtract threshold from ROI pixels
        """
        roi_mask, bkg_mask = self.compute_masks(circle)

        roi_means = []
        bkg_means = []
        net_means = []

        for img in self.fluorescent_images:
            roi_vals = img[roi_mask == 1]
            bkg_vals = img[bkg_mask == 1]

            if bkg_vals.size:
                # Log-transform background pixels
                log_bkg = np.log(bkg_vals + 1e-6)  # small offset to avoid log(0)
                mean_log = np.mean(log_bkg)
                std_log = np.std(log_bkg)

                # Noise threshold in original scale
                noise_thresh = np.exp(mean_log + 3 * std_log)
            else:
                noise_thresh = 0

            # Subtract threshold from ROI pixels
            roi_corrected = roi_vals - noise_thresh
            roi_corrected[roi_corrected < 0] = 0  # remove negative values

            roi_mean = float(np.mean(roi_vals)) if roi_vals.size else np.nan
            bkg_mean = float(np.mean(bkg_vals)) if bkg_vals.size else np.nan
            net_mean = float(np.mean(roi_corrected)) if roi_corrected.size else np.nan

            roi_means.append(roi_mean)
            bkg_means.append(bkg_mean)
            net_means.append(net_mean)

        # Update circle object
        circle.fluorescence = {
            ch: {"roi": roi, "bkg": bkg}
            for ch, roi, bkg in zip(self.channel_names, roi_means, bkg_means)
        }
        circle.net_fluorescence = {
            ch: net for ch, net in zip(self.channel_names, net_means)
        }

        return (
            dict(zip(self.channel_names, roi_means)),
            dict(zip(self.channel_names, bkg_means)),
            dict(zip(self.channel_names, net_means))
        )


    # ------------------------------------------
    # Extract fluorescence per circle
    # ------------------------------------------
    def extract_fluorescence(self, circle):
        """
        Extracts mean ROI and background values for all channels.
        Updates circle.fluorescence with per-channel values.
        Also computes net_fluorescence (ROI - Background) and stores in circle.
        """
        roi_mask, bkg_mask = self.compute_masks(circle)

        roi_means = []
        bkg_means = []
        net_means = []

        for img in self.fluorescent_images:
            roi_vals = img[roi_mask == 1]
            bkg_vals = img[bkg_mask == 1]

            roi_mean = float(roi_vals.mean()) if roi_vals.size else np.nan
            bkg_mean = float(bkg_vals.mean()) if bkg_vals.size else np.nan
            net_mean = roi_mean - bkg_mean if not np.isnan(roi_mean) and not np.isnan(bkg_mean) else np.nan

            roi_means.append(roi_mean)
            bkg_means.append(bkg_mean)
            net_means.append(net_mean)

        # Update circle object
        circle.fluorescence = {
            ch: {"roi": roi, "bkg": bkg}
            for ch, roi, bkg in zip(self.channel_names, roi_means, bkg_means)
        }
        circle.net_fluorescence = {
            ch: net for ch, net in zip(self.channel_names, net_means)
        }

        # Return for convenience
        return (
            dict(zip(self.channel_names, roi_means)),
            dict(zip(self.channel_names, bkg_means)),
            dict(zip(self.channel_names, net_means))
        )

    # ------------------------------------------
    # Compute global net mean across all circles
    # ------------------------------------------
    def compute_global_net_mean(self):
        """
        Computes global net mean per channel across all circles.
        Ignores NaN values.
        """
        if not self.circles:
            return {}

        channel_names = self.channel_names
        net_sums = {ch: 0.0 for ch in channel_names}
        n_counts = {ch: 0 for ch in channel_names}

        for circle in self.circles:
            if not hasattr(circle, "net_fluorescence"):
                continue
            for ch, net_val in circle.net_fluorescence.items():
                if not np.isnan(net_val):
                    net_sums[ch] += net_val
                    n_counts[ch] += 1

        global_net_means = {
            ch: (net_sums[ch] / n_counts[ch] if n_counts[ch] > 0 else np.nan)
            for ch in channel_names
        }
        return global_net_means

    # ------------------------------------------
    # Visualize ROIs
    # ------------------------------------------
    def visualize_rois(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.bf_image, cmap="gray")
        ax.set_title("ROI and Background Masks")
        ax.axis("equal")

        for circle in self.circles:
            roi_in  = circle.r + roi_inner
            roi_out = circle.r + roi_outer
            bkg_in  = circle.r + bckg_inner
            bkg_out = circle.r + bckg_outer

            # ROI rings (red)
            ax.add_patch(patches.Circle((circle.x, circle.y), roi_out,
                                        edgecolor='red', facecolor='none', linewidth=2))
            ax.add_patch(patches.Circle((circle.x, circle.y), roi_in,
                                        edgecolor='red', facecolor='none', linestyle='--'))

            # Background rings (blue)
            ax.add_patch(patches.Circle((circle.x, circle.y), bkg_out,
                                        edgecolor='blue', facecolor='none', linewidth=2))
            ax.add_patch(patches.Circle((circle.x, circle.y), bkg_in,
                                        edgecolor='blue', facecolor='none', linestyle='--'))

        # Add legend
        red_patch = patches.Patch(color='red', label='ROI')
        blue_patch = patches.Patch(color='blue', label='Background')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def compute_circle_and_global_net(circles):
    """
    Computes net fluorescence for each circle and global net means per channel.

    Updates each Circle object with:
        circle.net_fluorescence = {channel: roi - bkg}

    Returns:
        dict: {channel_name: global_net_mean} across all circles
    """
    if not circles:
        return {}

    channel_names = list(circles[0].fluorescence.keys())
    net_sums = {ch: 0.0 for ch in channel_names}
    n_counts = {ch: 0 for ch in channel_names}

    # Loop over each circle
    for circle in circles:
        circle.net_fluorescence = {}
        for ch, vals in circle.fluorescence.items():
            roi_val = vals['roi']
            bkg_val = vals['bkg']

            # Skip NaNs
            if np.isnan(roi_val) or np.isnan(bkg_val):
                circle.net_fluorescence[ch] = np.nan
                continue

            # Net per circle
            net_val = roi_val - bkg_val
            circle.net_fluorescence[ch] = net_val

            # Add to global sum
            net_sums[ch] += net_val
            n_counts[ch] += 1

    # Compute global net mean per channel
    global_net_means = {
        ch: (net_sums[ch] / n_counts[ch] if n_counts[ch] > 0 else np.nan)
        for ch in channel_names
    }

    return global_net_means

def remove_nearby_circles(circles, min_distance):
    """
    Remove circles that are too close to another circle.
    Keeps the first circle encountered and removes later ones.
    
    circles: list of Circle objects (must have .x and .y)
    min_distance: distance threshold for removal
    """
    pts = np.array([(c.x, c.y) for c in circles])
    tree = KDTree(pts)

    removed = set()
    kept = []

    for i, c in enumerate(circles):
        if i in removed:
            continue

        # Find neighbors within min_distance
        neighbors = tree.query_ball_point([c.x, c.y], min_distance)

        # Remove all except this one
        for n in neighbors:
            if n != i:
                removed.add(n)

        kept.append(c)

    return kept, [circles[i] for i in removed]

def remove_artifact_electrodes_global_threshold(circles, fl_img, roi_creator):
    """
    Implements the artifact-filtering method from the paper:
    
    - Compute global threshold = mean + 6*std across entire fluorescence image
    - For each circle, check BG pixels
    - If >0.1% of BG pixels exceed the threshold → remove the electrode
    """

    # --- 1. Global threshold ---
    img_vals = fl_img.flatten()
    global_mean = img_vals.mean()
    global_std = img_vals.std()
    global_threshold = global_mean + 6 * global_std

    valid = []
    removed = []

    for c in circles:
        _, bkg_mask = roi_creator.compute_masks(c)
        bkg_vals = fl_img[bkg_mask == 1]

        if bkg_vals.size == 0:
            removed.append(c)
            continue

        # Fraction of BG pixels above threshold
        frac_above = np.mean(bkg_vals > global_threshold)

        if frac_above <= 0.001:  # 99.9% must be below threshold
            valid.append(c)
        else:
            removed.append(c)

    return valid, removed, global_threshold

def check_artifacts_per_channel(circles, fl_images, roi_creator):
    """
    Multi-channel artifact detection.
    Removes electrodes if ANY channel fails the threshold test.

    Returns:
        valid_circles
        removed_circles
        artifact_details[channel][circle] = { pct_above_threshold, threshold, failed }
    """
    artifact_details = {}
    circles_to_remove = set()

    for ch_name, img in fl_images.items():

        # Extract BG pixels for each circle for this channel
        bg_pixel_data = []
        for c in circles:
            roi_mask, bg_mask = roi_creator.compute_masks(c)
            bg_pixels = img[bg_mask == 1]
            bg_pixel_data.append((c, bg_pixels))

        # Compute global threshold for this channel
        all_bg_pixels = np.concatenate([pix for _, pix in bg_pixel_data if len(pix) > 0])
        global_threshold = np.mean(all_bg_pixels) + 6 * np.std(all_bg_pixels)

        artifact_details[ch_name] = {}

        # Evaluate each circle against this channel's threshold
        for c, bg_pix in bg_pixel_data:

            if len(bg_pix) == 0:
                artifact_details[ch_name][c] = {
                    "pct_above_threshold": np.nan,
                    "threshold": global_threshold,
                    "failed": True
                }
                circles_to_remove.add(c)
                continue

            pct_above = np.sum(bg_pix > global_threshold) / len(bg_pix)
            failed = pct_above > 0.001   # 0.1% threshold from the paper

            artifact_details[ch_name][c] = {
                "pct_above_threshold": pct_above,
                "threshold": global_threshold,
                "failed": failed
            }

            if failed:
                circles_to_remove.add(c)

    valid = [c for c in circles if c not in circles_to_remove]
    removed = [c for c in circles if c in circles_to_remove]

    return valid, removed, artifact_details

def export_results_to_excel(
    output_folder,
    file_name,
    channel_names,
    global_means,
    all_circles =None,
    removed_artifacts=None,
    missing_circles=None
):
    """
    Writes analysis results into an Excel workbook with two sheets:
    1. Summary
    2. Electrode data (per circle, per channel)
    """
    # ---------------------------------------------------------
    # SHEET 1 — SUMMARY TABLE
    # ---------------------------------------------------------
    summary_rows = [{
        "file": file_name,
        **{ch: global_means.get(ch, np.nan) for ch in channel_names},
        "Electrodes Used" : len(all_circles) if all_circles is not None else 0,
        "Electrodes Added": len(missing_circles) if missing_circles is not None else 0,
        "Electrodes Removed": len(removed_artifacts) if removed_artifacts is not None else 0,
    }]

    df_summary = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # SHEET 2 — ELECTRODE TABLE
    # ---------------------------------------------------------
    electrode_rows = []

    for circle in all_circles:

        base_info = {
            "index": circle.index,
            "x": circle.x,
            "y": circle.y,
            "radius": circle.r,
        }

        # Add per-channel values
        for ch in channel_names:

            # ROI and background (already stored)
            if ch in circle.fluorescence:
                roi_val = circle.fluorescence[ch]["roi"]
                bkg_val = circle.fluorescence[ch]["bkg"]
            else:
                roi_val = np.nan
                bkg_val = np.nan

            # Net fluorescence (your new DFQ-style)
            net_val = circle.net_fluorescence.get(ch, np.nan)

            base_info[f"{ch}_ROI"] = roi_val
            base_info[f"{ch}_Background"] = bkg_val
            base_info[f"{ch}_Net"] = net_val

        electrode_rows.append(base_info)

    df_electrodes = pd.DataFrame(electrode_rows)

    # ---------------------------------------------------------
    # WRITE EXCEL FILE
    # ---------------------------------------------------------
    excel_path = os.path.join(output_folder, f"{file_name}_results.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_electrodes.to_excel(writer, sheet_name="Electrodes", index=False)

    print(f"\nSaved results → {excel_path}\n")


# ------------------------------------------ Optional Visualizations ------------------------------------------

def plot_pixel_histogram(circle, img, roi_mask, bkg_mask, channel_name='Channel 1'):
    """
    Plots histogram of ROI and BG pixels with threshold for a single electrode.
    
    circle: Circle object
    img: 2D NumPy array (fluorescence image)
    roi_mask: 2D array mask for ROI
    bkg_mask: 2D array mask for BG
    method: 'local_threshold' or 'mean_subtract'
    channel_name: string label for the channel
    """
    roi_vals = img[roi_mask == 1]
    bkg_vals = img[bkg_mask == 1]

    # Compute threshold
    if bkg_vals.size:
        log_bkg = np.log(bkg_vals + 1e-6)
        mean_log = np.mean(log_bkg)
        std_log = np.std(log_bkg)
        threshold = np.exp(mean_log + 3*std_log)
    else:
        threshold = np.mean(bkg_vals) if bkg_vals.size else 0

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(roi_vals, bins=30, alpha=0.7, color='green', label='ROI pixels')
    plt.hist(bkg_vals, bins=30, alpha=0.5, color='blue', label='BG pixels')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    
    plt.xlabel('Fluorescence intensity')
    plt.ylabel('Pixel count')
    plt.title(f'Pixel intensity histogram - {channel_name} (Electrode {circle.index})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_kde_with_threshold(circle, img, roi_mask, bkg_mask, channel_name='Channel 1', show = True):
    """
    Plots smoothed density (KDE) of ROI and BG pixels with threshold.

    circle: Circle object
    img: 2D NumPy array (fluorescence image)
    roi_mask: 2D array mask for ROI
    bkg_mask: 2D array mask for background
    method: 'local_threshold' or 'mean_subtract'
    channel_name: string label for the channel
    show: whether to call plt.show()
    """
    roi_vals = img[roi_mask == 1]
    bkg_vals = img[bkg_mask == 1]

    # Compute threshold
    if bkg_vals.size:
        log_bkg = np.log(bkg_vals + 1e-6)
        threshold = np.exp(np.mean(log_bkg) + 3*np.std(log_bkg))
    else:
        threshold = np.mean(bkg_vals) if bkg_vals.size else 0

    plt.figure(figsize=(6,4))
    sns.kdeplot(roi_vals, fill=True, color='green', alpha=0.5, label='ROI')
    sns.kdeplot(bkg_vals, fill=True, color='blue', alpha=0.3, label='BG')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')

    plt.xlabel('Fluorescence intensity')
    plt.ylabel('Density')
    plt.title(f'KDE - {channel_name} (Electrode {circle.index})')
    plt.legend()
    plt.tight_layout()
    if show:    
        plt.show()

def plot_violin_with_threshold(circle, img, roi_mask, bkg_mask, channel_name='Channel 1', show = True):
    """
    Violin plot comparing ROI vs BG pixels with threshold.
    """
    roi_vals = img[roi_mask == 1]
    bkg_vals = img[bkg_mask == 1]

    # Compute threshold
    if bkg_vals.size:
        log_bkg = np.log(bkg_vals + 1e-6)
        threshold = np.exp(np.mean(log_bkg) + 3*np.std(log_bkg))
    else:
        threshold = np.mean(bkg_vals) if bkg_vals.size else 0

    data = pd.DataFrame({
        'Intensity': np.concatenate([roi_vals, bkg_vals]),
        'Region': ['ROI']*len(roi_vals) + ['BG']*len(bkg_vals)
    })

    plt.figure(figsize=(5,4))
    sns.violinplot(x='Region', y='Intensity', data=data, palette={'ROI':'green','BG':'blue'})
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    plt.title(f'Violin - {channel_name} (Electrode {circle.index})')
    plt.tight_layout()
    if show:
        plt.show()

def plot_removed_electrodes(bf_img, valid, removed):
    plt.figure(figsize=(10, 10))
    plt.imshow(bf_img, cmap='gray')
    ax = plt.gca()

    # Plot valid electrodes in green
    for c in valid:
        circ = patches.Circle((c.x, c.y), c.r, edgecolor='lime', fill=False, linewidth=2)
        ax.add_patch(circ)

    # Plot removed electrodes in red
    for c in removed:
        circ = patches.Circle((c.x, c.y), c.r, edgecolor='red', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.text(c.x, c.y, "X", color='red', fontsize=12, ha='center', va='center')

    plt.title("Valid (Green) vs Removed (Red) Electrodes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_average_electrode_3d(circles, img, roi_creator, channel_name="unknown", smooth_sigma=2):
    """
    Create one averaged 'super-electrode' 3D profile by combining all electrodes.
    """
    roi_list = []
    max_r = int(max([c.r for c in circles]))

    for c in circles:
        x, y, r = int(c.x), int(c.y), int(c.r)

        x0, x1 = x - r, x + r
        y0, y1 = y - r, y + r

        # Crop ROI
        roi = img[y0:y1, x0:x1]

        # Skip if empty
        if roi is None or roi.size == 0:
            continue

        # Resize to a consistent shape (2r × 2r)
        roi_resized = cv2.resize(roi, (2*max_r, 2*max_r), interpolation=cv2.INTER_LINEAR)
        roi_list.append(roi_resized)

    if len(roi_list) == 0:
        print("No valid ROIs to average.")
        return

    # Average all ROIs
    avg_roi = np.mean(np.stack(roi_list, axis=0), axis=0)

    # Optional smoothing
    if smooth_sigma > 0:
        avg_roi = gaussian_filter(avg_roi, sigma=smooth_sigma)

    # Create grid for X,Y
    h, w = avg_roi.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # 3D Plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, avg_roi, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_title(f"Averaged Electrode Fluorescence (Channel: {channel_name})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()

def plot_multi_channel_3d(circles, fl_images_dict, roi_creator, smooth_sigma=2):
    """
    Overlay multiple channels in 3D on one averaged electrode with fixed channel colors.
    circles: list of valid Circle objects
    fl_images_dict: {channel_name: image_array}
    roi_creator: your ROIcreation instance
    """

    max_r = int(max([c.r for c in circles]))

    # Fixed channel-to-color mapping
    channel_colors = {
        "DsRed": "red",
        "EGFP": "green",
        "DAPI": "blue",
        "Cy5": "magenta"
    }

    # Prepare figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    legend_handles = []

    for ch_name, img in fl_images_dict.items():
        if img is None:
            print(f"[3D plot] Skipping channel {ch_name} — image is None")
            continue

        roi_list = []

        for c in circles:
            x, y, r = int(c.x), int(c.y), int(c.r)
            x0, x1 = x - r, x + r
            y0, y1 = y - r, y + r

            # Skip out-of-bounds
            if y0 < 0 or y1 > img.shape[0] or x0 < 0 or x1 > img.shape[1]:
                continue

            roi = img[y0:y1, x0:x1]
            if roi is None or roi.size == 0:
                continue

            # Resize to consistent shape
            roi_resized = cv2.resize(roi, (2*max_r, 2*max_r), interpolation=cv2.INTER_LINEAR)
            roi_list.append(roi_resized)

        if len(roi_list) == 0:
            print(f"No valid ROIs for channel {ch_name}")
            continue

        # Average and smooth
        avg_roi = np.mean(np.stack(roi_list, axis=0), axis=0)
        if smooth_sigma > 0:
            avg_roi = gaussian_filter(avg_roi, sigma=smooth_sigma)

        # Create grid
        h, w = avg_roi.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # Get color from mapping, default to gray if not listed
        color = channel_colors.get(ch_name, "gray")

        # Plot surface
        surf = ax.plot_surface(
            X, Y, avg_roi,
            color=color,
            alpha=0.5,
            linewidth=0,
            antialiased=True
        )

        # Keep handle for legend
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=ch_name))

    # Add legend
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_title("Overlay of Fluorescence Channels (Averaged Electrodes)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Intensity")
    plt.tight_layout()
    plt.show()


# ------------------------------------------ MAIN SCRIPT ------------------------------------------------------

if __name__ == "__main__":
    tif_or_czi = input("Analyze TIF or CZI files? (t/c): ").strip().lower()

    finder = FindFolders(tif_or_czi)
    parent_dir, folders = finder.run()
    print(f"Folders or files: {folders}")

    for f in tqdm(folders, desc='Processing Folders', colour='green', leave=True): # --- This part iterates over each tiff or czi file that is inside of the parent directory --- 
        organizer = FileOrganization(f, tif_or_czi)
        organization_result = organizer.run()

        if debug:
            print(f"Organized: {organization_result}")

        loader = UniversalImageLoader(tif_or_czi)

        # --- TIFF workflow ---
        if tif_or_czi == "t":
            # Load brightfield
            brightfield_image = loader.load(organization_result["brightfield"])
            bf_8bit = cv2.normalize(brightfield_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_h, img_w = bf_8bit.shape[:2]
            output_folder = organization_result["output_folder"]

            # Load fluorescent images
            fl_images = loader.load_fluorescent_images(
                fl_image_files=organization_result["fluorescent"]
            )

        # --- CZI workflow ---
        elif tif_or_czi == "c":
            # Build channel map
            channel_map = {name: i for i, name in enumerate(organization_result["channels"])}

            # Load brightfield
            bf_idx = channel_map.get("RL Brightfield")
            brightfield_image = loader.load(organization_result["czi_file"], channel_index=bf_idx) if bf_idx is not None else None
            bf_8bit = cv2.normalize(brightfield_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if brightfield_image is not None else None
            img_h, img_w = bf_8bit.shape[:2]
            output_folder = organization_result["output_folder"]

            # Load fluorescent images safely
            fl_images = loader.load_fluorescent_images(
                czi_file=organization_result["czi_file"],
                channel_map=channel_map
            )
        
        
        circle_detector = CircleDetection(bf_8bit)
        result = circle_detector.run()  

        # Check what was returned
        if "bd" in settings.lower():
            outer_circles, inner_circles = result, []
            outer_circles, removed_boundaries = filter_circles_inside_image(outer_circles, img_w, img_h)
            rows, sorted_circles = sort_circles(outer_circles)
        else:
            outer_circles, inner_circles = result
            inner_circles, removed_boundaries = filter_circles_inside_image(inner_circles, img_w, img_h)
            rows, sorted_circles = sort_circles(inner_circles)

        # ---------- Find missing (using spacing from current rows) ----------
        spacing_values = [b.x - a.x for row in rows for a, b in zip(row, row[1:])]
        match_tolerance = int(0.20 * statistics.median(spacing_values))

        min_remove = match_tolerance 
        valid_circles, removed_circles = remove_nearby_circles(sorted_circles, min_remove)


        rows, sorted_circles = sort_circles(valid_circles)

        missing_positions = find_missing_circles(rows, match_tolerance=match_tolerance)
        missing_circles = []

        if missing_positions:
            med_r = statistics.median([c.r for c in sorted_circles])
            for mx, my in missing_positions:
                missing_circles.append(Circle(mx, my, med_r, parent=None, index=None))

        # ---------- Combine and re-sort ----------
        all_circles = sorted_circles + missing_circles
        rows, all_circles = sort_circles(all_circles)

        channel_names = []
        fl_image_list = []

        for ch, img in fl_images.items():
            if img is not None:
                channel_names.append(ch)
                fl_image_list.append(img)

        roi_creator = ROIcreation(all_circles, fl_image_list, channel_names, bf_8bit)

        # --- NEW MULTI-CHANNEL ARTIFACT REMOVAL ---
        fl_images_dict = {name: img for name, img in zip(channel_names, fl_image_list)}

        valid_circles, removed_artifacts, artifact_info = check_artifacts_per_channel(
            all_circles,
            fl_images_dict,
            roi_creator
        )

        print(f"\n=== Artifact Removal Summary Across All Channels ===")
        print(f"Removed {len(removed_artifacts)} electrodes.\n")

        for c in removed_artifacts:
            print(f"Electrode index={c.index}, x={c.x:.1f}, y={c.y:.1f} FAILED on channels:")
            for ch in artifact_info:
                if artifact_info[ch][c]["failed"]:
                    pct = artifact_info[ch][c]["pct_above_threshold"] * 100
                    thr = artifact_info[ch][c]["threshold"]
                    print(f"   • {ch}: {pct:.4f}% BG pixels above threshold ({thr:.2f})")


        # Continue with only valid
        all_circles = valid_circles
        rows, all_circles = sort_circles(all_circles)


        # --- ROI extraction already done ---
        for circle in all_circles:
            #roi_creator.extract_fluorescence(circle) # Original method of roi avg - background avg
            roi_creator.extract_fluorescence_local_threshold(circle) # closer to DFQ method of subtracting local threshold

            roi_mask, bkg_mask = roi_creator.compute_masks(circle)

        global_means = compute_circle_and_global_net(all_circles)
        print(f"\nGlobal Means: {global_means}")


        export_results_to_excel(
            output_folder=output_folder,
            file_name=os.path.basename(f).replace(".czi", "").replace(".tif", ""),
            channel_names=channel_names,
            global_means=global_means,
            all_circles=all_circles,
            removed_artifacts = removed_artifacts,
            missing_circles = missing_circles
        )

        # ------------------------ DISPLAY ---------------------------
        if display:
            display_circles(bf_8bit, outer_circles, inner_circles)
            disp_removed_circles(bf_8bit, removed_circles)
            disp_missing_circles(bf_8bit, outer_circles, inner_circles, missing_circles)
            disp_all_circles(bf_8bit, all_circles)
            roi_creator.visualize_rois(None)
            # Skip None images
            valid_fl_images = {ch: img for ch, img in fl_images.items() if img is not None}
            # Single call for multi-channel 3D plot
            plot_multi_channel_3d(valid_circles, valid_fl_images, roi_creator, smooth_sigma=2)
                    # Visual map
            plot_removed_electrodes(bf_8bit, valid_circles, removed_artifacts)

        # ------------------------ DEBUG --------------------------------
        if debug:
            if bf_8bit is not None:
                print(f"Brightfield image shape: {bf_8bit.shape}, dtype: {bf_8bit.dtype}")
            else:
                print("No brightfield image found.")

            for key, img in fl_images.items():
                if img is not None:
                    print(f"{key} image shape: {img.shape}, dtype: {img.dtype}")
                else:
                    print(f"{key} channel not present in this file.")
            
            print(f"Detected {len(outer_circles)} outer circles")
            for c in outer_circles:
                        print(f"Outer circle: x={c.x}, y={c.y}, r={c.r}")

            print(f"Detected {len(inner_circles)} inner circles")
            for c in inner_circles:
                print(f"Inner circle: x={c.x}, y={c.y}, r={c.r}, parent_radius={c.parent.r}")
            
            print(f"\n Removed Circles Outside of Image Boundaries: ")
            for c in removed_boundaries:
                reason = []
                if c.x - c.r < 0:    reason.append("left edge")
                if c.x + c.r >= img_w:  reason.append("right edge")
                if c.y - c.r < 0:    reason.append("top edge")
                if c.y + c.r >= img_h:  reason.append("bottom edge")
                reason = ", ".join(reason)
                
                idx = getattr(c, "index", None)
                idx_str = f"Circle {idx}" if idx is not None else "(unindexed circle)"

                print(f"{idx_str}: (x={c.x:.1f}, y={c.y:.1f}, r={c.r:.1f}) → {reason}")

            print(f"Using Match Tolerance: {match_tolerance}")

            print(f"Removed Circles: {len(removed_circles)}, {removed_circles}")

            print(f"Missing Circles: {len(missing_circles)}, {missing_circles}")

            print(f"Sorted Circles:")
            for c in all_circles:
                parent_r = c.parent.r if c.parent is not None else None
                print(f"\nCircle {c.index}: x={c.x}, y={c.y}, r={c.r}, parent_radius={parent_r}")

            for c in all_circles:
                    print(f"\nCircle {c.index}: x={c.x}, y={c.y}, r={c.r}")
                    for ch, vals in c.fluorescence.items():
                        print(f"  {ch}: ROI={vals['roi']:.2f}, Background={vals['bkg']:.2f}")

            print("-------------------------------------------------------")
            for c in all_circles:
                print(f"\nNet Fl {c.index}: x={c.x}, y={c.y}, r={c.r}, net_fl={c.net_fluorescence}")

            print(f"Global Means: {global_means}")
            print("-------------------------------------------------------")

end = time.time()
print(f"Program Completed in : {end - start} s") 
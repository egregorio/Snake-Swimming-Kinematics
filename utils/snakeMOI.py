# Import libraries
import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString, Point
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist


import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree

def calculate_moment_of_inertia_voronoi(binary_mask, centerline):
    """
    Estimate moment of inertia using Voronoi-style partitioning from centerline segments.

    Parameters:
        binary_mask (2D array): Binary image mask of the object.
        centerline (Nx2 array): Ordered centerline coordinates (x, y).

    Returns:
        float: Estimated moment of inertia in pixels^4.
    """

    # 1. Compute center of mass
    com_y, com_x = center_of_mass(binary_mask)
    com = np.array([com_x, com_y])

    # 2. Compute centerline segment midpoints
    midpoints = []
    for i in range(len(centerline) - 1):
        p1 = centerline[i]
        p2 = centerline[i + 1]
        mid = (p1 + p2) / 2
        midpoints.append(mid)
    midpoints = np.array(midpoints)

    if len(midpoints) == 0:
        return 0.0  # nothing to compute

    # 3. Get all pixels inside the object
    ys, xs = np.where(binary_mask > 0)
    pixels = np.stack([xs, ys], axis=1)  # shape (N, 2)

    # 4. Assign each pixel to nearest midpoint
    tree = cKDTree(midpoints)
    _, labels = tree.query(pixels)

    # 5. Accumulate moment of inertia
    total_I = 0.0
    for i in range(len(midpoints)):
        region_pixels = pixels[labels == i]
        if len(region_pixels) == 0:
            continue
        region_center = np.mean(region_pixels, axis=0)
        d = np.linalg.norm(region_center - com)
        area = len(region_pixels)
        total_I += area * (d ** 2)

    return total_I, com


def convert_inertia_pixels_to_meters(I_pixels, pixel_length_px, reference_mm=25):
    """
    Convert moment of inertia from pixels^4 to meters^4 using a known scale.

    Parameters:
        I_pixels (float): Moment of inertia in pixel^4.
        pixel_length_px (float): Length in pixels that corresponds to reference_mm.
        reference_mm (float): Physical length in millimeters (default is 25mm).

    Returns:
        float: Moment of inertia in meters^4.
    """
    pixel_size_m = (reference_mm / pixel_length_px) / 1000  # convert mm to meters
    
    return I_pixels * (pixel_size_m ** 4)

def convert_geometric_to_physical_inertia(I_geometric_m4, binary_mask, pixel_size_meters, body_mass_kg):
    """
    Convert geometric moment of inertia (in m^4) to physical moment of inertia (kg·m^2),
    assuming uniform mass distribution over the object.

    Parameters:
        I_geometric_m4 (float): Geometric moment of inertia (in m^4).
        binary_mask (2D array): Binary mask of the object.
        pixel_size_meters (float): Size of one pixel in meters.
        body_mass_kg (float): Total mass of the object in kilograms.

    Returns:
        float: Physical moment of inertia (in kg·m^2).
    """
    body_mass_kg = body_mass_kg / 1000

    pixel_area_m2 = pixel_size_meters ** 2
    total_area_m2 = np.sum(binary_mask > 0) * pixel_area_m2

    if total_area_m2 == 0:
        raise ValueError("Binary mask has zero area.")

    surface_density = body_mass_kg / total_area_m2  # kg/m²
    I_physical = I_geometric_m4 * surface_density   # kg·m²

    return I_physical



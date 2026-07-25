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

def estimateWidths(contours, smoothed_centerline):
    """
    Estimates the width of the snake at every point along the centerline.
    
    Parameters:
        smoothed_centerline (np.ndarray): Nx2 array of [x, y] centerline coordinates.
        contours (np.ndarray): Nx2 array of [x,y] points that give the snake outline.
        
    Returns:
        widths (list or np.ndarray): Widths along the centerline.
        half_lengths (list or np.ndarray): Half widths along the centerline.
    """
    
    # Get the contour as an Nx2 array
    cnt = contours[0][:, 0, :]

    # Create a Shapely Polygon from the outline
    snake_polygon = Polygon(cnt)

    # Store widths and endpoints
    widths = []
    half_lengths = []

    for i in range(1, len(smoothed_centerline) - 1):
        # Current and neighboring points
        p_prev = smoothed_centerline[i - 1]
        p_curr = smoothed_centerline[i]
        p_next = smoothed_centerline[i + 1]

        # Tangent vector
        tangent = p_next - p_prev
        if np.linalg.norm(tangent) == 0:
            widths.append(0)
            half_lengths.append((None, None))
            continue

        tangent = tangent / np.linalg.norm(tangent)

        # Perpendicular (normal) vector
        normal = np.array([-tangent[1], tangent[0]])

        # Extend in both directions from the centerline
        extension = 100  # pixels
        p1 = p_curr + normal * extension
        p2 = p_curr - normal * extension

        # Create line and find intersection with polygon boundary
        perp_line = LineString([p1, p2])
        intersections = perp_line.intersection(snake_polygon.boundary)

        # Handle multiple cases
        pts = []
        if intersections.geom_type == 'MultiPoint':
            pts = list(intersections.geoms)
        elif intersections.geom_type == 'GeometryCollection':
            pts = [g for g in intersections.geoms if g.geom_type == 'Point']
        elif intersections.geom_type == 'Point':
            pts = [intersections]

        if len(pts) >= 2:
            pts = sorted(pts, key=lambda pt: pt.distance(Point(p_curr)))
            width = pts[0].distance(pts[1])
            widths.append(width)
            half_lengths.append((pts[0], pts[1]))
        else:
            widths.append(0)
            half_lengths.append((None, None))

    return widths, half_lengths

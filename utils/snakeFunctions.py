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

def findSnake(image):
    """
    Takes an image of a snake and converts it into a binary image and finds the outline and skeleton of the snake.
    
    Parameters:
        img (image): image of the snake.
    Returns:
        binary (np.ndarray): Binary image mask of the object.
        contours (np.ndarray): Nx2 array of [x,y] points that give the snake outline.
        skeleton (np.ndarray): Nx2 binary array of the snake's skeleton.
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(~image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert binary image to boolean (True for foreground)
    binary_bool = binary > 0

    # Skeletonize (centerline)
    skeleton = skeletonize(binary_bool)

    return binary, contours, skeleton

def find_endpoints(skeleton):
    """
    Finds the endpoints for the snake's skeleton.
    
    Parameters:
        skeleton (np.ndarray): Nx2 binary array of the snake's skeleton.
    Returns:
        endpoints (np.ndarray): Nx2 array of (x,y) coordinates of the skeleton's end points.
    """

    y, x = np.where(skeleton)
    
    # initialize list for endpoints
    endpoints = []

    # 8-connected neighborhood
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])

    for xi, yi in zip(x, y):
        roi = skeleton[yi-1:yi+2, xi-1:xi+2]
        if roi.shape == (3,3):
            val = np.sum(roi * kernel)
            if val == 11:  # pixel itself + 1 neighbor = 11
                endpoints.append((xi, yi))
    
    return endpoints

def trace_skeleton(skel_img, start):
    """
    Puts the (x,y) points of the found skeleton in the correct order from start to end
    
    Parameters:
        skel_img (image): boolean image of the snake's skeleton.
        start (np.ndarray): (x,y) coordinate of the skeleton's starting point
    Returns:
        smoothed_centerline (np.ndarray): Nx2 array of [x, y] centerline coordinates.
    """

    visited = set()
    path = [start]
    current = start

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),         ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    while True:
        visited.add(current)
        x, y = current
        found = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= ny < skel_img.shape[0]) and (0 <= nx < skel_img.shape[1]):
                if skel_img[ny, nx] and (nx, ny) not in visited:
                    path.append((nx, ny))
                    current = (nx, ny)
                    found = True
                    break
        if not found:
            break
            
    return np.array(path)

def smoothCenterline(ordered_skeleton):
    """
    Smooths the centerline of the snake.
    
    Parameters:
        ordered_skeleton (np.ndarray): Nx2 array of [x,y] ordered points of snake centerline.
        
    Returns:
        smoothed_centerline (np.ndarray): Nx2 array of [x, y] centerline coordinates.
    """
    
    # Convert to float arrays
    x = ordered_skeleton[:, 0]
    y = ordered_skeleton[:, 1]

    # Fit a B-spline (s=0 means no smoothing, you can increase it slightly if needed)
    tck, u = splprep([x, y], s=5)

    # Sample N evenly spaced points along the spline
    N = 200  # You can change this number
    u_new = np.linspace(0, 1, N)
    x_smooth, y_smooth = splev(u_new, tck)

    # Stack into array of points for later use
    smoothed_centerline = np.stack((x_smooth, y_smooth), axis=-1)

    return smoothed_centerline



# square_detection.py
import numpy as np
import cv2

# Global image dimensions
COLOR_WIDTH, COLOR_HEIGHT = 640, 480

def clamp_point(pt, width=COLOR_WIDTH, height=COLOR_HEIGHT):
    """
    Clamp (x, y) coordinates so that they lie within the image boundaries.
    """
    x = max(0, min(width - 1, pt[0]))
    y = max(0, min(height - 1, pt[1]))
    return (int(x), int(y))

def find_squares(image, min_area=1000):
    """
    Detect square-like contours in an image and ignore ones with small area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area < min_area:
                continue
            squares.append(approx)
    return squares

def order_points(pts, marker_angle=0):
    """
    Order square corners in the following order: 
    top-left, top-right, bottom-right, bottom-left.
    If marker_angle is provided, rotate the ordering so that the first edge matches the marker's direction.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    if marker_angle:
        best_edge = get_rotation_edge(rect, marker_angle)
        start_idx = None
        for i in range(4):
            if np.allclose(rect[i], best_edge[0]):
                start_idx = i
                break
        if start_idx is not None and start_idx != 0:
            rect = np.vstack((rect[start_idx:], rect[:start_idx]))
    return rect

def get_rotation_edge(ordered_corners, marker_angle=0):
    """
    Given the ordered corners of a square and a marker angle (in degrees),
    compute the edge that best aligns with the arrow's direction.
    """
    center = np.mean(ordered_corners, axis=0)
    theta = np.radians(marker_angle)
    arrow_dir = np.array([np.cos(theta), np.sin(theta)])
    edges = [(ordered_corners[i], ordered_corners[(i + 1) % 4]) for i in range(4)]
    
    best_edge = None
    best_dot = -1
    for edge in edges:
        p1 = np.array(edge[0])
        p2 = np.array(edge[1])
        midpoint = (p1 + p2) / 2.0
        vec = midpoint - center
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec_normalized = vec / norm
        dot = np.dot(arrow_dir, vec_normalized)
        if dot > best_dot:
            best_dot = dot
            best_edge = edge
    return best_edge

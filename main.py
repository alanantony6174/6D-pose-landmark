import pyrealsense2 as rs
import numpy as np
import cv2
import json
from pose import PoseEstimator
from detector import YOLOOBBDetector  # Import the YOLO detector
from rotation import RotationMarkerEstimator  # Import the rotation marker estimator

# Global image dimensions (as used in the pipeline)
COLOR_WIDTH, COLOR_HEIGHT = 640, 480

# ------------------------------
# Helper functions in main.py
# ------------------------------
def clamp_point(pt, width=COLOR_WIDTH, height=COLOR_HEIGHT):
    """Clamp (x, y) coordinates so that they lie within the image boundaries."""
    x = max(0, min(width - 1, pt[0]))
    y = max(0, min(height - 1, pt[1]))
    return (int(x), int(y))

def find_squares(image, min_area=1000):
    """Detect square-like contours in an image and ignore ones with small area."""
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

def order_points(pts):
    """
    Order square corners in the following order: 
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_rotation_edge(ordered_corners, marker_angle=0):
    """
    Given the ordered corners of a square and a marker angle in degrees,
    compute the edge that best aligns with the arrow's direction.
    
    This implementation computes the square's center and, for each edge,
    the midpoint of that edge. It then calculates the dot product between
    the normalized arrow direction vector and the normalized vector from
    the center to the midpoint. The edge with the maximum dot product is chosen.
    """
    # Compute the center of the square.
    center = np.mean(ordered_corners, axis=0)
    
    # Convert marker angle to radians and compute the arrow's unit vector.
    theta = np.radians(marker_angle)
    arrow_dir = np.array([np.cos(theta), np.sin(theta)])
    
    # Define edges as pairs of consecutive points.
    edges = [(ordered_corners[i], ordered_corners[(i + 1) % 4]) for i in range(4)]
    
    best_edge = None
    best_dot = -1  # Dot product ranges from -1 to 1.
    
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

# =====================================
# Initialize RealSense Pipeline
# =====================================
pipeline = rs.pipeline()
config = rs.config()

color_width, color_height = COLOR_WIDTH, COLOR_HEIGHT
depth_width, depth_height = 640, 480
fps = 30

config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is:", depth_scale)

color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
print("Camera Intrinsics:", color_intrinsics)

# Create an instance of the PoseEstimator with the camera intrinsics.
pose_estimator = PoseEstimator(color_intrinsics)

# Initialize the YOLO detector.
model_path = "models/alpha-1.pt"  # Adjust this path if needed
detector = YOLOOBBDetector(model_path)

# Initialize the Rotation Marker Estimator.
# Adjust the reference image path if needed.
ref_img_path = "AlphadroidTM_logo-01_Super_Tool.jpg"
rotation_estimator = RotationMarkerEstimator(model_path, ref_img_path)

print("Starting stream. Detecting YOLO bounding boxes, computing squares inside them, and overlaying pose.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Get the original color image.
        color_image = np.asanyarray(color_frame.get_data())
        # Create a copy for drawing overlays (detections, pose, etc.).
        display_image = color_image.copy()

        # Get YOLO detections on the full color image.
        detections = detector.predict_enlarged_bbox(color_image, scale_factor=1.3)
        for det in detections:
            # Each detection provides a quadrilateral bbox.
            bbox_poly = det["bbox"]  # e.g., shape (4, 1, 2)
            name = det["name"]
            conf = det["confidence"]

            # (Optional) Draw the YOLO bounding box in red.
            # tl_point = tuple(bbox_poly[0][0])
            # cv2.putText(display_image, f"{name} {conf:.2f}",
            #             (tl_point[0], tl_point[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Compute axis-aligned bounding rectangle from the quadrilateral.
            x, y, w, h = cv2.boundingRect(bbox_poly)

            # Crop the region of interest (ROI) from the color image.
            roi = color_image[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Detect squares inside the ROI using the find_squares function.
            squares = find_squares(roi, min_area=1000)
            if len(squares) > 0:
                # Process only the first square found in the ROI.
                square = squares[0]
                # Adjust square coordinates back to the full image space.
                square += np.array([[[x, y]]])
                cv2.polylines(display_image, [square], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Order the corners: [top-left, top-right, bottom-right, bottom-left]
                ordered_corners = order_points(square)
                for pt in ordered_corners:
                    cv2.circle(display_image, clamp_point(pt), 4, (255, 0, 0), -1)

                # Compute pose for the detected square.
                try:
                    pose_info = pose_estimator.estimate_pose(ordered_corners, depth_frame, axis_length=0.1)
                except Exception as e:
                    print("Pose estimation error:", e)
                    continue

                plane_normal = pose_info['plane_normal']
                rotation_matrix = pose_info['rotation_matrix']
                quaternion = pose_info['quaternion']
                endpoints_2d = pose_info['endpoints_2d']

                print("Fitted Plane Normal:", plane_normal)
                print("Rotation Matrix:\n", rotation_matrix)
                print("Quaternion (x, y, z, w):", quaternion)
                text = f"Quat: {np.round(quaternion, 2)}"
                cv2.putText(display_image, text, clamp_point((10, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

                # Draw the pose axes overlay.
                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['x']),
                         (0, 0, 255), 2)
                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['y']),
                         (0, 255, 0), 2)
                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['z']),
                         (255, 0, 0), 2)
                cv2.circle(display_image, clamp_point(endpoints_2d['center']), 3, (255, 255, 255), -1)
                
                # Get the marker angle from the rotation estimator.
                # This returns only the angle in degrees (or None if detection fails).
                marker_angle = rotation_estimator.process_image(color_image)
                if marker_angle is None:
                    marker_angle = 0  # Fallback if no valid angle is detected

                # Draw the edge in the direction of the detected marker angle.
                rotation_edge = get_rotation_edge(ordered_corners, marker_angle)
                cv2.line(display_image, clamp_point(rotation_edge[0]), clamp_point(rotation_edge[1]), (0, 255, 255), 2)
                cv2.putText(display_image, f"Marker: {marker_angle:.1f} deg", clamp_point((rotation_edge[0][0], rotation_edge[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # If you wish to process only the first valid detection, you can break here.
                # break

        # Display the resulting image.
        cv2.imshow("RealSense with YOLO ROI", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except Exception as e:
    print("Exception occurred:", e)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

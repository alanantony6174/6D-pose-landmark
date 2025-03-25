import pyrealsense2 as rs
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
#import matplotlib.pyplot as plt  # for 3D plotting
#from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # for drawing polygons in 3D
 
#plt.ion()  # enable interactive mode for matplotlib
 
# Global image dimensions (as used in the pipeline)
COLOR_WIDTH, COLOR_HEIGHT = 640, 480
 
# ------------------------------------
# Helper function: Clamp point to image bounds
# ------------------------------------
def clamp_point(pt, width=COLOR_WIDTH, height=COLOR_HEIGHT):
    """Clamp (x, y) coordinates so that they lie within the image boundaries."""
    x = max(0, min(width - 1, pt[0]))
    y = max(0, min(height - 1, pt[1]))
    return (int(x), int(y))
 
# ------------------------------------
# Helper function: Square detection
# ------------------------------------
def find_squares(image):
    """Detect square-like contours in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            squares.append(approx)
    return squares
 
# ------------------------------------
# Helper function: Order square corners
# Returns points in order: top-left, top-right, bottom-right, bottom-left
# ------------------------------------
def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
 
# ------------------------------------
# Helper function: Get average depth over a window
# ------------------------------------
def get_average_depth(depth_frame, x, y, window_size=3):
    """Return the average depth value around (x, y) over a window."""
    half = window_size // 2
    depths = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x_idx = int(round(x)) + i
            y_idx = int(round(y)) + j
            if x_idx < 0 or y_idx < 0:
                continue
            depth = depth_frame.get_distance(x_idx, y_idx)
            if depth > 0:
                depths.append(depth)
    return np.mean(depths) if depths else depth_frame.get_distance(int(round(x)), int(round(y)))
 
# ================================
# Initialize RealSense Pipeline
# ================================
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
 
print("Starting stream. Continuously detecting square, computing pose and showing overlay.")
 
# Create a 3D figure for visualization.
# fig_3d = plt.figure("3D Visualization")
# ax_3d = fig_3d.add_subplot(111, projection='3d')
 
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
 
        color_image = np.asanyarray(color_frame.get_data())
        squares = find_squares(color_image)
        for square in squares:
            cv2.polylines(color_image, [square], isClosed=True, color=(0, 255, 0), thickness=2)
 
        if len(squares) > 0:
            # Use the first detected square and order its corners.
            square = squares[0]
            ordered_corners = order_points(square)  # [tl, tr, br, bl]
            for pt in ordered_corners:
                cv2.circle(color_image, clamp_point(pt), 4, (255, 0, 0), -1)
            tl, tr, br, bl = ordered_corners
 
            # ----- Sample points along the edges of the square -----
            num_points_edge = 8  # sample 8 points per edge (including corners)
            sample_points = []
            # Top edge: from tl to tr
            for i in range(num_points_edge):
                u = i / (num_points_edge - 1)
                x = (1 - u) * tl[0] + u * tr[0]
                y = (1 - u) * tl[1] + u * tr[1]
                depth = depth_frame.get_distance(int(round(x)), int(round(y)))
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
                sample_points.append({"x": point_3d[0], "y": point_3d[1], "z": point_3d[2]})
            # Right edge: from tr to br
            for i in range(num_points_edge):
                u = i / (num_points_edge - 1)
                x = (1 - u) * tr[0] + u * br[0]
                y = (1 - u) * tr[1] + u * br[1]
                depth = depth_frame.get_distance(int(round(x)), int(round(y)))
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
                sample_points.append({"x": point_3d[0], "y": point_3d[1], "z": point_3d[2]})
            # Bottom edge: from br to bl
            for i in range(num_points_edge):
                u = i / (num_points_edge - 1)
                x = (1 - u) * br[0] + u * bl[0]
                y = (1 - u) * br[1] + u * bl[1]
                depth = depth_frame.get_distance(int(round(x)), int(round(y)))
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
                sample_points.append({"x": point_3d[0], "y": point_3d[1], "z": point_3d[2]})
            # Left edge: from bl to tl
            for i in range(num_points_edge):
                u = i / (num_points_edge - 1)
                x = (1 - u) * bl[0] + u * tl[0]
                y = (1 - u) * bl[1] + u * tl[1]
                depth = depth_frame.get_distance(int(round(x)), int(round(y)))
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth)
                sample_points.append({"x": point_3d[0], "y": point_3d[1], "z": point_3d[2]})
 
            # ----- Compute deprojected 3D corners using averaged depth -----
            tl_depth = get_average_depth(depth_frame, tl[0], tl[1], window_size=3)
            tr_depth = get_average_depth(depth_frame, tr[0], tr[1], window_size=3)
            br_depth = get_average_depth(depth_frame, br[0], br[1], window_size=3)
            bl_depth = get_average_depth(depth_frame, bl[0], bl[1], window_size=3)
            tl_3d = np.array(rs.rs2_deproject_pixel_to_point(color_intrinsics, [tl[0], tl[1]], tl_depth))
            tr_3d = np.array(rs.rs2_deproject_pixel_to_point(color_intrinsics, [tr[0], tr[1]], tr_depth))
            br_3d = np.array(rs.rs2_deproject_pixel_to_point(color_intrinsics, [br[0], br[1]], br_depth))
            bl_3d = np.array(rs.rs2_deproject_pixel_to_point(color_intrinsics, [bl[0], bl[1]], bl_depth))
 
            # ----- Define the square plane exactly using the four corners -----
            # Compute the plane normal from two edges (tl->tr and tl->bl)
            plane_normal = np.cross(tr_3d - tl_3d, bl_3d - tl_3d)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            if plane_normal[2] < 0:
                plane_normal = -plane_normal
 
            # Define x-axis as the normalized top edge (from tl to tr)
            x_axis = tr_3d - tl_3d
            x_axis = x_axis / np.linalg.norm(x_axis)
            # Define y-axis as the cross product of plane_normal and x_axis.
            y_axis = np.cross(plane_normal, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            # Compute center of the square as the average of the four corners.
            center_3d = (tl_3d + tr_3d + br_3d + bl_3d) / 4
 
            # Compute rotation matrix and quaternion (columns: x_axis, y_axis, plane_normal)
            rotation_matrix = np.column_stack((x_axis, y_axis, plane_normal))
            quaternion = R.from_matrix(rotation_matrix).as_quat()
            print("Fitted Plane Normal:", plane_normal)
            print("Rotation Matrix:\n", rotation_matrix)
            print("Quaternion (x, y, z, w):", quaternion)
            text = f"Quat: {np.round(quaternion, 2)}"
            cv2.putText(color_image, text, clamp_point((10, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
 
            # ----- Draw 2D Pose Overlay -----
            axis_length = 0.1  # in meters; adjust as needed
            center_2d = rs.rs2_project_point_to_pixel(color_intrinsics, center_3d.tolist())
            x_endpoint_3d = (center_3d + x_axis * axis_length).tolist()
            x_endpoint_2d = rs.rs2_project_point_to_pixel(color_intrinsics, x_endpoint_3d)
            y_endpoint_3d = (center_3d + y_axis * axis_length).tolist()
            y_endpoint_2d = rs.rs2_project_point_to_pixel(color_intrinsics, y_endpoint_3d)
            z_endpoint_3d = (center_3d + plane_normal * axis_length).tolist()
            z_endpoint_2d = rs.rs2_project_point_to_pixel(color_intrinsics, z_endpoint_3d)
            pose_image = color_image.copy()
            cv2.line(pose_image, clamp_point(center_2d), clamp_point(x_endpoint_2d),
                     (0, 0, 255), 2)
            cv2.line(pose_image, clamp_point(center_2d), clamp_point(y_endpoint_2d),
                     (0, 255, 0), 2)
            cv2.line(pose_image, clamp_point(center_2d), clamp_point(z_endpoint_2d),
                     (255, 0, 0), 2)
            cv2.circle(pose_image, clamp_point(center_2d), 3, (255, 255, 255), -1)
            # Also draw the top edge (yellow) for verification.
            cv2.line(pose_image, clamp_point(tl), clamp_point(tr), (0, 255, 255), 2)
            cv2.putText(pose_image, "Top Edge", clamp_point((tl[0], tl[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Pose", pose_image)
 
        cv2.imshow("RealSense + Square Detection", color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
 
except Exception as e:
    print("Exception occurred:", e)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# main.py
import pyrealsense2 as rs
import numpy as np
import cv2
from square import clamp_point, find_squares, order_points, get_rotation_edge
from pose import PoseEstimator
from detector import YOLOOBBDetector  # YOLO detector
from rotation import RotationMarkerEstimator  # Rotation marker estimator

# Global image dimensions (as used in the pipeline)
COLOR_WIDTH, COLOR_HEIGHT = 640, 480

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
detector = YOLOOBBDetector(model_path, )

# Initialize the Rotation Marker Estimator.
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
        display_image = color_image.copy()

        # Get YOLO detections on the full color image.
        detections = detector.predict_enlarged_bbox(color_image, scale_factor=1.3)
        for det in detections:
            bbox_poly = det["bbox"]
            name = det["name"]
            conf = det["confidence"]

            # Compute axis-aligned bounding rectangle.
            x, y, w, h = cv2.boundingRect(bbox_poly)
            roi = color_image[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Use square detection functions from the module.
            squares = find_squares(roi, min_area=1000)
            if len(squares) > 0:
                square = squares[0]
                square += np.array([[[x, y]]])
                cv2.polylines(display_image, [square], isClosed=True, color=(0, 255, 0), thickness=2)
                
                marker_angle = rotation_estimator.process_image(color_image)
                if marker_angle is None:
                    marker_angle = 0

                ordered_corners = order_points(square, marker_angle)
                for pt in ordered_corners:
                    cv2.circle(display_image, clamp_point(pt), 4, (255, 0, 0), -1)

                try:
                    pose_info = pose_estimator.estimate_pose(ordered_corners, depth_frame, axis_length=0.1)
                except Exception as e:
                    print("Pose estimation error:", e)
                    continue

                endpoints_2d = pose_info['endpoints_2d']
                rotation_edge = get_rotation_edge(ordered_corners, marker_angle)
                cv2.line(display_image, clamp_point(rotation_edge[0]), clamp_point(rotation_edge[1]), (0, 255, 255), 2)
                cv2.putText(display_image, f"Marker: {marker_angle:.1f} deg", 
                            clamp_point((rotation_edge[0][0], rotation_edge[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                text = f"Quat: {np.round(pose_info['quaternion'], 2)}"
                cv2.putText(display_image, text, clamp_point((10, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['x']),
                         (0, 0, 255), 2)
                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['y']),
                         (0, 255, 0), 2)
                cv2.line(display_image, clamp_point(endpoints_2d['center']), clamp_point(endpoints_2d['z']),
                         (255, 0, 0), 2)
                cv2.circle(display_image, clamp_point(endpoints_2d['center']), 3, (255, 255, 255), -1)

                # Optionally, process only the first valid detection.
                # break

        cv2.imshow("RealSense with YOLO ROI", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except Exception as e:
    print("Exception occurred:", e)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

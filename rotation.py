import cv2
import numpy as np
import math
from ultralytics import YOLO
from detector import YOLOOBBDetector

class RotationMarkerEstimator:
    def __init__(self, model_path, ref_img_path):
        # Initialize YOLO Detector
        self.detector = YOLOOBBDetector(model_path)

        # Load the Reference Image in Grayscale
        self.ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        if self.ref_img is None:
            raise ValueError("Error: Reference image not found at path: " + ref_img_path)
        
        # Compute SIFT Features on the Reference Image
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ref_img, None)
        if self.des1 is None or len(self.kp1) == 0:
            raise ValueError("Error: No descriptors found in the reference image!")
        
        # Set up FLANN Matcher for SIFT Matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def process_image(self, color_image):
        """
        Processes the input color image, computes the rotation of the detected object,
        and returns the marker angle in degrees.

        If no valid detection or sufficient feature matches are found, returns None.
        """
        # YOLO Object Detection to get ROI; scale factor can be adjusted as needed
        detections = self.detector.predict_enlarged_bbox(color_image, scale_factor=1.25)
        if len(detections) == 0:
            # If no detection, return None
            return None
        
        detection = detections[0]
        roi_polygon = detection["bbox"]  # Expected shape: (N, 1, 2)
        
        # Compute ROI centroid (if needed for further processing)
        roi_points = roi_polygon.reshape(-1, 2)
        centroid = np.mean(roi_points, axis=0)
        
        # Convert image to grayscale for SIFT detection
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.sift.detectAndCompute(gray_frame, None)
        if des2 is None or len(kp2) == 0:
            return None
        
        # Filter keypoints/descriptors to those inside the ROI
        filtered_kp2 = []
        filtered_des2 = []
        for kp, desc in zip(kp2, des2):
            if cv2.pointPolygonTest(roi_polygon, kp.pt, False) >= 0:
                filtered_kp2.append(kp)
                filtered_des2.append(desc)
        
        if len(filtered_kp2) < 5:
            return None
        
        kp2 = filtered_kp2
        des2 = np.array(filtered_des2)
        
        # Match features between the reference image and the live ROI
        matches = self.flann.knnMatch(self.des1, des2, k=2)
        good_matches = []
        angle_diffs = []  # To store the angular differences
        
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                # Use SIFT keypoint angles (in degrees)
                ref_angle = self.kp1[m.queryIdx].angle
                live_angle = kp2[m.trainIdx].angle
                # Compute difference and normalize to [0, 360)
                diff = (live_angle - ref_angle) % 360
                angle_diffs.append(diff)
        
        if len(good_matches) < 5:
            return None
        
        # Compute Circular Mean of Angle Differences
        angle_diffs_rad = [np.deg2rad(a) for a in angle_diffs]
        sum_sin = np.sum([np.sin(a) for a in angle_diffs_rad])
        sum_cos = np.sum([np.cos(a) for a in angle_diffs_rad])
        mean_angle_rad = math.atan2(sum_sin, sum_cos)
        # Normalize mean angle to [0, 360)
        mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
        
        return mean_angle_deg

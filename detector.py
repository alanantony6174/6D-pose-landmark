from ultralytics import YOLO
import numpy as np

class YOLOOBBDetector:
    def __init__(self, model_path: str):
        """
        Initializes the detector by loading the YOLO model.
        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def predict_enlarged_bbox(self, image, scale_factor: float = 1.25):
        """
        Processes an input image and returns a list of detections containing only the enlarged bounding boxes.
        :param image: Input image (numpy array, BGR format).
        :param scale_factor: Factor by which to enlarge the detected bounding boxes.
        :return: List of detections. Each detection is a dictionary with:
                 'name': class name,
                 'bbox': enlarged bounding box points (numpy array of shape (N, 1, 2)),
                 'confidence': confidence score.
        """
        results = self.model(image)
        detections = []

        for result in results:
            for obb in result.obb:
                # Ensure (N,2) shape for the polygon points.
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                conf = obb.conf.item()
                cls = int(obb.cls.item())
                name = result.names[cls]

                # Compute center of the polygon
                center_x = np.mean(xyxyxyxy[:, 0])
                center_y = np.mean(xyxyxyxy[:, 1])

                # Compute enlarged polygon points
                enlarged_points = []
                for x, y in xyxyxyxy:
                    new_x = int(center_x + (x - center_x) * scale_factor)
                    new_y = int(center_y + (y - center_y) * scale_factor)
                    enlarged_points.append([new_x, new_y])
                enlarged_points = np.array(enlarged_points, dtype=int).reshape(-1, 1, 2)

                detections.append({
                    "name": name,
                    "bbox": enlarged_points,
                    "confidence": conf
                })
        return detections

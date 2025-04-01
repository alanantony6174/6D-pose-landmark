import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

class PoseEstimator:
    def __init__(self, color_intrinsics):
        """
        Initialize the PoseEstimator with the camera's color intrinsics.
        """
        self.color_intrinsics = color_intrinsics

    def _get_average_depth(self, depth_frame, x, y, window_size=3):
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
        if depths:
            return np.mean(depths)
        else:
            return depth_frame.get_distance(int(round(x)), int(round(y)))

    def estimate_pose(self, ordered_corners, depth_frame, axis_length=0.1, window_size=3):
        """
        Estimate the 3D pose given 4 ordered corners (top-left, top-right,
        bottom-right, bottom-left) and the depth frame.
        
        Returns:
            A dictionary with the following keys:
              - center_3d: 3D center point of the square.
              - plane_normal: normal vector of the square's plane.
              - rotation_matrix: 3x3 rotation matrix (columns: x_axis, y_axis, plane_normal).
              - quaternion: pose in quaternion format (x, y, z, w).
              - endpoints_2d: 2D projections of the x, y, z axes endpoints for overlay.
        """
        # Unpack corners
        tl, tr, br, bl = ordered_corners

        # Compute averaged depths for each corner
        tl_depth = self._get_average_depth(depth_frame, tl[0], tl[1], window_size)
        tr_depth = self._get_average_depth(depth_frame, tr[0], tr[1], window_size)
        br_depth = self._get_average_depth(depth_frame, br[0], br[1], window_size)
        bl_depth = self._get_average_depth(depth_frame, bl[0], bl[1], window_size)

        # Deproject 2D pixels to 3D points.
        tl_3d = np.array(rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [tl[0], tl[1]], tl_depth))
        tr_3d = np.array(rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [tr[0], tr[1]], tr_depth))
        br_3d = np.array(rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [br[0], br[1]], br_depth))
        bl_3d = np.array(rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [bl[0], bl[1]], bl_depth))

        # Compute plane normal from two edges.
        plane_normal = np.cross(tr_3d - tl_3d, bl_3d - tl_3d)
        norm_plane = np.linalg.norm(plane_normal)
        if norm_plane < 1e-6:
            raise ValueError("Degenerate plane: normal vector is too small.")
        plane_normal = plane_normal / norm_plane
        if plane_normal[2] < 0:
            plane_normal = -plane_normal

        # Define x-axis as the normalized top edge (from tl to tr)
        x_axis = tr_3d - tl_3d
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            raise ValueError("Degenerate edge: top edge length is too small.")
        x_axis = x_axis / norm_x

        # Define y-axis as the cross product of plane_normal and x_axis.
        y_axis = np.cross(plane_normal, x_axis)
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            raise ValueError("Degenerate axis: y-axis length is too small.")
        y_axis = y_axis / norm_y

        # Compute center of the square as the average of the four corners.
        center_3d = (tl_3d + tr_3d + br_3d + bl_3d) / 4

        # Build rotation matrix (columns: x_axis, y_axis, plane_normal).
        rotation_matrix = np.column_stack((x_axis, y_axis, plane_normal))
        try:
            quaternion = R.from_matrix(rotation_matrix).as_quat()
        except Exception as e:
            raise ValueError("Quaternion computation failed: " + str(e))

        # Compute endpoints for pose axes (for overlay purposes).
        x_endpoint_3d = (center_3d + x_axis * axis_length).tolist()
        y_endpoint_3d = (center_3d + y_axis * axis_length).tolist()
        z_endpoint_3d = (center_3d + plane_normal * axis_length).tolist()

        x_endpoint_2d = rs.rs2_project_point_to_pixel(self.color_intrinsics, x_endpoint_3d)
        y_endpoint_2d = rs.rs2_project_point_to_pixel(self.color_intrinsics, y_endpoint_3d)
        z_endpoint_2d = rs.rs2_project_point_to_pixel(self.color_intrinsics, z_endpoint_3d)
        center_2d = rs.rs2_project_point_to_pixel(self.color_intrinsics, center_3d.tolist())

        endpoints_2d = {
            'center': center_2d,
            'x': x_endpoint_2d,
            'y': y_endpoint_2d,
            'z': z_endpoint_2d
        }

        return {
            'center_3d': center_3d,
            'plane_normal': plane_normal,
            'rotation_matrix': rotation_matrix,
            'quaternion': quaternion,
            'endpoints_2d': endpoints_2d
        }

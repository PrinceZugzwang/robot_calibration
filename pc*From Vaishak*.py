"""
This script provides a 3D point cloud visualizer that is view only/ is the base class for the PointCloudController.

Example usage:
    ```
    # View-only mode (no cut execution):
    python pointcloud.py
    ```
"""

import argparse
import os
from typing import Any, Optional, Set

import open3d as o3d
import trimesh

# Default file paths defined as constants
CUT_TRAJECTORY_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/owl_teleop/owl_teleop/fk_data/fk_trajectory_20250327-122421.npy"
DEFAULT_GLB_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/mast3r/pointclouds/state_2.glb"
GRASP_TRAJECTORY_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/owl_teleop/owl_teleop/reference_traj/motion_data_20250402-125755.npy"
DEFAULT_CAMERA_POSES_PATH = "/mnt/02D0BBD8D0BBCFE1/repos/owl_teleop/owl_teleop/calibration/config/camera_poses.npy"


class PointCloud:
    def __init__(
        self,
        glb_path: str = DEFAULT_GLB_PATH,
    ) -> None:
        """
        Initialize the PointCloud with configuration settings.

        Args:
            cut (bool): Whether to enable cutting functionality. Defaults to False.
            grasp (bool): Whether to enable grasp functionality. Defaults to False.
            glb_path (str): Path to the GLB file to visualize. Defaults to DEFAULT_GLB_PATH.
            calibrate (bool): Whether to enable calibration functionality. Defaults to False.

        Returns:
            None
        """
        self.points = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.vis: Optional[o3d.visualization.VisualizerWithVertexSelection] = None
        self.selected_points: Set[Any] = set()
        self.glb_path: str = glb_path

    def load_glb(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        Load a GLB file and convert it into an Open3D point cloud.

        This method:
        1. Loads the GLB file using trimesh
        2. Extracts vertices as points and colors
        3. Creates and configures an Open3D point cloud

        Args:
            file_path (str): Path to the GLB file to load

        Returns:
            o3d.geometry.PointCloud: The loaded and configured point cloud object
        """
        # Load the GLB file using trimesh
        scene = trimesh.load(file_path)
        points = scene.geometry["geometry_0"].vertices
        print(points)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = scene.geometry["geometry_0"].colors[:, :3]
        # Convert from 0-255 to 0-1
        colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd = pcd
        return pcd

    def create_coordinate_frame(self, size: float = 1.0) -> o3d.geometry.TriangleMesh:
        """
        Create a coordinate frame for reference in the visualization.

        This creates a 3D coordinate frame with XYZ axes shown in red, green, and blue,
        positioned at the origin to help with spatial orientation.

        Args:
            size (float): The size of the coordinate frame. Defaults to 1.0.

        Returns:
            o3d.geometry.TriangleMesh: The created coordinate frame mesh
        """
        # Create a coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0]
        )
        return coordinate_frame

    def run(self) -> Set[Any]:
        """
        Run the point cloud visualization and interaction session.

        This method:
        1. Loads the specified GLB file
        2. Sets up the visualization window and adds the point cloud
        3. Configures the selection callback if enabled
        4. Adds a coordinate frame for reference
        5. Runs the visualization loop until user exits

        Args:
            None (uses class attributes)

        Returns:
            Set[Any]: A set of selected points (if any were selected)
        """
        # Load the GLB file
        if not os.path.exists(self.glb_path):
            print(f"Error: File not found at {self.glb_path}")
            return set()

        point_cloud = self.load_glb(self.glb_path)

        # Create a visualization window
        self.vis = o3d.visualization.VisualizerWithVertexSelection()
        if self.vis is None:
            print("Error: Failed to create visualizer")
            return set()

        self.vis.create_window()

        # Add the point cloud to the visualizer
        self.vis.add_geometry(point_cloud)

        # Add coordinate frame
        coordinate_frame = self.create_coordinate_frame()
        self.vis.add_geometry(coordinate_frame)

        # Set the default camera view
        self.vis.get_view_control().set_zoom(0.8)

        # Run the visualizer
        self.vis.run()

        # Clean up
        self.vis.destroy_window()

        return self.selected_points


def main() -> None:
    """
    This function:
    1. Parses command line arguments
    2. Initializes the PointCloudVisualizer with the specified settings
    3. Runs the visualization session
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Point Cloud Visualizer")
    parser.add_argument(
        "--glb-path",
        type=str,
        default=DEFAULT_GLB_PATH,
        help=f"Path to the GLB file (default: {DEFAULT_GLB_PATH})",
    )
    args = parser.parse_args()

    # Initialize visualizer with selection enabled/disabled based on argument
    visualizer = PointCloud(glb_path=args.glb_path)
    visualizer.run()


if __name__ == "__main__":
    main()
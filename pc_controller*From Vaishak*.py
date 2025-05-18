"""
This script provides a 3D point cloud visualizer that allows users to:
    - Load and visualize GLB files containing 3D point cloud data
    - Interactively select points in the visualization
    - Execute cutting trajectories at selected points
    - View the point cloud in a 3D space with coordinate frame reference

Example usage:
    ```
    # View-only mode (no cut execution):
    python pointcloud.py

    # Enable cut execution on point selection:
    python pointcloud.py --execute-cut

    # Specify custom trajectory and GLB files:
    python pointcloud.py --execute-cut --trajectory-path path/to/trajectory.npy --glb-path path/to/model.glb
    ```
"""

import argparse
import os
from typing import Any, Set

import open3d as o3d

from owl_teleop.calibration.get_transform_mat import apply_transform_matrix
from owl_teleop.control.utils import exectute_traj, transform_traj
from owl_teleop.utils.pointcloud import (
    CUT_TRAJECTORY_PATH,
    DEFAULT_GLB_PATH,
    GRASP_TRAJECTORY_PATH,
    PointCloud,
)


class PointCloudController(PointCloud):
    def __init__(
        self,
        cut: bool = False,
        grasp: bool = False,
        glb_path: str = DEFAULT_GLB_PATH,
        calibrate: bool = False,
    ) -> None:
        super().__init__(glb_path)

        self.cut: bool = cut
        self.grasp: bool = grasp
        self.calibrate: bool = calibrate

    def cut_callback(self) -> bool:
        """
        Callback function that is triggered when a point is selected in the visualization.

        This method:
        1. Gets the most recently selected point's coordinates
        2. Applies a transformation matrix to convert to robot space
        3. Transforms the trajectory to go through the selected point
        4. Executes the cutting operation at that point in a separate thread

        Args:
            None (uses class attributes)

        Returns:
            bool: True to indicate the callback completed successfully
        """
        if self.vis is None:
            print("Error: Visualizer is not initialized")
            return False

        picked_points = self.vis.get_picked_points()
        if not picked_points:
            print("Error: No points were picked")
            return False

        transformed_point = apply_transform_matrix(picked_points[0].coord)
        transformed_traj = transform_traj(CUT_TRAJECTORY_PATH, transformed_point)

        print("Executing cut trajectory")
        exectute_traj(transformed_traj, "127.0.0.1", ik=True, gripper=False, port=12346)

        return True

    def grasp_callback(self) -> bool:
        """
        Callback function that is triggered when a point is selected in the visualization.

        This method executes the grasp trajectory in a separate thread.
        """
        print("Executing grasp trajectory")
        exectute_traj(
            GRASP_TRAJECTORY_PATH, "127.0.0.1", ik=False, gripper=True, port=12345
        )

        return True

    def calibrate_callback(self) -> bool:
        """
        Callback function that is triggered when a point is selected in the visualization.

        This method retrieves the selected point coordinates for calibration purposes.

        Returns:
            bool: True to indicate the callback completed successfully
        """
        print("calibrate callback")
        if self.vis is None:
            print("Error: Visualizer is not initialized")
            return False

        picked_points = self.vis.get_picked_points()
        if not picked_points:
            print("Error: No points were picked")
            return False

        picked_point = picked_points[0].coord
        print("picked point", picked_point)
        return True

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

        if self.calibrate:
            self.vis.register_selection_changed_callback(self.calibrate_callback)
            print(
                "Selection callback enabled. Click on points to execute calibrate there."
            )
        elif self.cut:
            self.vis.register_selection_changed_callback(self.cut_callback)
            print("Selection callback enabled. Click on points to execute cuts there.")
        elif self.grasp:
            self.vis.register_selection_changed_callback(self.grasp_callback)
            print(
                "Selection callback enabled. Click on points to execute grasps there."
            )
        else:
            print("Selection callback disabled. Running in view-only mode.")

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
        "--execute-cut",
        action="store_true",
        help="Enable point selection callback for executing cuts (default: disabled)",
    )
    parser.add_argument(
        "--trajectory-path",
        type=str,
        default=CUT_TRAJECTORY_PATH,
        help=f"Path to the trajectory file (default: {CUT_TRAJECTORY_PATH})",
    )
    parser.add_argument(
        "--glb-path",
        type=str,
        default=DEFAULT_GLB_PATH,
        help=f"Path to the GLB file (default: {DEFAULT_GLB_PATH})",
    )
    parser.add_argument(
        "--execute-grasp",
        action="store_true",
        help="Enable point selection callback for executing grasps (default: disabled)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable point selection callback for executing calibrate (default: disabled)",
    )
    args = parser.parse_args()

    # Initialize visualizer with selection enabled/disabled based on argument
    visualizer = PointCloudController(
        cut=args.execute_cut,
        grasp=args.execute_grasp,
        glb_path=args.glb_path,
        calibrate=args.calibrate,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
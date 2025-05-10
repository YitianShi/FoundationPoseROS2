from pathlib import Path
import open3d as o3d
import numpy as np
import json

def depth_to_pcd(depth_image, intrinsic_matrix):
    """
    Convert a depth image to a point cloud.

    Args:
        depth_image (numpy.ndarray): Depth image (H, W) with depth values in meters.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix (3, 3).

    Returns:
        open3d.geometry.PointCloud: Point cloud object.
    """
    # Create a meshgrid of pixel coordinates
    depth_image = np.asarray(depth_image)
    h, w = depth_image.height, depth_image.width
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    z = depth_image.flatten()

    # Convert pixel coordinates to normalized device coordinates
    x = (u - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0] * z
    y = (v - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1] * z

    # Stack the coordinates into a point cloud
    points = np.vstack((x, y, z)).T

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

if __name__ == "__main__":
    data_path = Path("images_20250415_103256").absolute()
    color_image = o3d.io.read_image(data_path / "rgb" / "image_0123.jpg")  # Load your color image
    width, height = np.asarray(color_image).shape[:2]
    depth_image = o3d.io.read_image(data_path / "depth" / "image_0123.png")  # Load your depth image
    with open(data_path / "camera_intrinsics.json", "r") as f:
        intrinsic_matrix = json.load(f)  # Load your intrinsic matrix
    intrinsic_matrix = np.array(intrinsic_matrix["intrinsic_matrix"]).reshape(3, 3)
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        intrinsic_matrix=intrinsic_matrix
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_matrix
    )


    #pcd = depth_to_pcd(depth_image, intrinsic_matrix)

    # Save the point cloud to a file
    o3d.io.write_point_cloud("output.pcd", pcd)
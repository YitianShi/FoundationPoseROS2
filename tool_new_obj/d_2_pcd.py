import numpy as np
import open3d as o3d
import imageio
import os
import glob

def depth_to_point_cloud(depth, intrinsic_matrix, depth_scale=1000.0, pcd_image=False):
    if isinstance(depth, str):
        depth = imageio.imread(depth).astype(np.float32) / depth_scale
    height, width = depth.shape

    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    points = np.stack((x3, y3, z3), axis=-1).reshape(-1, 3)

    # Remove zero-depth points
    # points = points[z3.flatten() > 0]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Flip the coordinate system
    pcd.transform([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    # Back to original image shape
    if pcd_image:
        return points.reshape(height, width, 3)
    return pcd

    

if __name__ == "__main__":
    depth_folder = "depth/"  # Folder containing depth PNGs
    output_folder = "point_cloud/"  # Output folder for PLY files
    os.makedirs(output_folder, exist_ok=True)

    K = np.array([[748.52520751953120,   0.0, 636.4364624023438],
                  [  0.0, 748.3948364257812, 351.7748718261719],
                  [  0.0,   0.0,   1.0]])

    depth_files = glob.glob(os.path.join(depth_folder, "*.png"))

    for depth_path in depth_files:
        filename = os.path.splitext(os.path.basename(depth_path))[0]
        ply_path = os.path.join(output_folder, f"{filename}.ply")
        pcd = depth_to_point_cloud(depth_path, K, ply_path)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Saved: {ply_path}")


import trimesh
from pathlib import Path
import numpy as np

def create_trimesh_scene():
    path = Path('images_20250416_103900/point_cloud/point_cloud_0030.ply')
    # Create a scene
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis())
    # Load a PointCloud
    point_cloud = trimesh.load(path)
    # Add the PointCloud to the scene
    scene.add_geometry(point_cloud)
    # Set the camera position
    scene.camera_transform = np.loadtxt(path.parent.parent / 'camera_poses' / Path(path.stem.replace('point_cloud', 'image')).with_suffix('.txt'))

    scene.show()

if __name__ == "__main__":
    create_trimesh_scene()
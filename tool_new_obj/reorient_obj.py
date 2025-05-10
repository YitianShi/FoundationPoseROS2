import trimesh
import numpy as np
from sklearn.decomposition import PCA

# Step 1: Load mesh
mesh = trimesh.load('demo_data/model/model.obj')

# Step 2: Extract vertices
vertices = mesh.vertices.copy()

# Step 3: Center the mesh at its center of mass
center_of_mass = vertices.mean(axis=0)
vertices_centered = vertices - center_of_mass

# Step 4: PCA to find principal axes
pca = PCA(n_components=3)
pca.fit(vertices_centered)
rotation_matrix = pca.components_

# Ensure right-handed coordinate system (optional but usually helpful)
if np.linalg.det(rotation_matrix) < 0:
    rotation_matrix[2, :] *= -1

# Step 5: Apply rotation
rotated_vertices = vertices_centered @ rotation_matrix.T

# Step 6: Update mesh and save
mesh.vertices = rotated_vertices
mesh.export('demo_data/model/aligned_model.obj')


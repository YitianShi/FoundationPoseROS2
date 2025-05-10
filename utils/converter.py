import pickle
import numpy as np
import trimesh

# Load the data from the .pkl file
with open('_DATA/data/mano/MANO_RIGHT.pkl', 'rb') as file:
    data = pickle.load(file)

# Assuming the data contains vertices and faces
vertices = np.array(data['vertices'])
faces = np.array(data['faces'])

# Create the mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Save the mesh to a file (e.g., .obj format)
mesh.export('output_mesh.obj')

# Optionally, visualize the mesh
mesh.show()

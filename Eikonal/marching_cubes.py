import os
import random
import argparse
import numpy as np
import colorsys
from skimage import measure
import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_network

# For extracting mesh through marching cubes:
# python marching_cubes.py --name output --epochs 2000


########################
# this function you need to complete
def compute_SDFs(net, device, nb_grid):
    x = np.linspace(-1.5, 1.5, nb_grid)
    y = np.linspace(-1.5, 1.5, nb_grid)
    z = np.linspace(-1.5, 1.5, nb_grid)
    X, Y, Z = np.meshgrid(x, y, z)
    
    ...

    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', '-n', type=str, default='output', help='output model name')

    args = parser.parse_args()
    name = args.name
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = build_network(input_dim=3)
    net.to(device)
    net.load_state_dict(torch.load('./models/{}_model.pth'.format(name), map_location=device))

    nb_grid = 128
    val = compute_SDFs(net, device, nb_grid)
    volume = val.reshape(nb_grid, nb_grid, nb_grid)
    
    verts, faces, normals, values = measure.marching_cubes(volume, 0.0, spacing=(1.0, -1.0, 1.0), gradient_direction='ascent')
    
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output', exist_ok=True)
    o3d.io.write_triangle_mesh("output/{}_mesh.ply".format(name), mesh)

import os
import random
import argparse
import numpy as np
import colorsys
from skimage import measure
import skimage
import plyfile, logging
import tqdm
from shapes import *

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_grid2D(shape2D, ranges=((0., 1.), (0., 1.)), flatten=True):
    x_dim = shape2D[1]
    y_dim = shape2D[0]
    x_range = ranges[0][0] - ranges[0][1]; y_range = ranges[1][0] - ranges[1][1]
    x_lin = np.linspace(ranges[0][0], ranges[0][1], x_dim, endpoint=False) + x_range / x_dim * 0.5
    y_lin = np.linspace(ranges[1][0], ranges[1][1], y_dim, endpoint=False) + y_range / y_dim * 0.5
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    if not flatten:
        return x_grid, y_grid
    x_t = torch.from_numpy(x_grid).cuda().float()
    y_t = torch.from_numpy(y_grid).cuda().float()
    x_flat = torch.reshape(x_t, [x_dim*y_dim, 1])
    y_flat = torch.reshape(y_t, [x_dim*y_dim, 1])
    xy = torch.cat((x_flat, y_flat), 1)
    return xy

def compute_sdf_2D_slice_unet(feature_flat, grid_trans, dec, zslice, grid2D, resol):
    # feature_flat: feature volume result after the UNet CNN, flattened
    xy_coord = grid2D.unsqueeze(0).cuda()
    xyz_coord = torch.cat([xy_coord, torch.ones(1,xy_coord.shape[1],1).cuda()*zslice], 2)
    grid_trans.overlap = False
    xy_code_ind, xy_rel_coord = grid_trans.compute_index(xyz_coord)
    z_selected = torch.cat([torch.index_select(feature_flat[ii], 1, spat_ind).unsqueeze(0)
                            for ii, spat_ind in enumerate(xy_code_ind)], 0)
    z_selected = z_selected.permute([0, 2, 1])
    input_all = torch.cat([xy_rel_coord, z_selected], 2)
    sdf_pred = dec(input_all).squeeze(-1)
    sdf_im = sdf_pred.detach().reshape([resol[0], resol[1]])
    return sdf_im.cpu()

def compute_sdf_3D_unet(feature_flat, grid_trans, dec, resol, ranges):
    xy_flat = compute_grid2D(resol, ranges=ranges)
    z_dim = resol[2]
    z_range = ranges[2][1] - ranges[2][0]
    z_lin = np.linspace(ranges[2][0], ranges[2][1], z_dim, endpoint=False) + z_range / z_dim * 0.5
    print(z_lin[-1])
    volume = torch.zeros([resol[2], resol[1], resol[0]])
    for z_ind, z_val in enumerate(z_lin):
        sdf_im = compute_sdf_2D_slice_unet(feature_flat, grid_trans, dec, z_val, xy_flat, resol)
        volume[z_ind] = sdf_im
    return volume

def compute_sdf_3D_gen(codebook, decoder, resol, ranges):
    xy_flat = compute_grid2D(resol, ranges=ranges).unsqueeze(0).cuda()
    z_dim = resol[2]
    z_range = ranges[2][1] - ranges[2][0]
    z_lin = np.linspace(ranges[2][0], ranges[2][1], z_dim, endpoint=False) + z_range / z_dim * 0.5
    volume = torch.zeros([resol[2], resol[1], resol[0]])
    for z_ind, z_val in enumerate(z_lin):
        xyz_coord = torch.cat([xy_flat, torch.ones(1, xy_flat.shape[1], 1).cuda() * z_val], 2)
        sdf_pred = GridSamplerDecoder.distributed_forward(decoder, codebook, xyz_coord).detach()
        sdf_im = sdf_pred.reshape([resol[0], resol[1]])
        volume[z_ind] = sdf_im
    return volume

def compute_sdf_3D_deepsdf(decoder, code, resol, ranges):
    # code: C
    xy_flat = compute_grid2D(resol, ranges=ranges).unsqueeze(0).cuda()
    z_dim = resol[2]
    z_range = ranges[2][1] - ranges[2][0]
    z_lin = np.linspace(ranges[2][0], ranges[2][1], z_dim, endpoint=False) + z_range / z_dim * 0.5
    volume = torch.zeros([resol[2], resol[1], resol[0]])
    for z_ind, z_val in enumerate(z_lin):
        xyz_coord = torch.cat([xy_flat, torch.ones(1, xy_flat.shape[1], 1).cuda() * z_val], 2)
        code_repeat = code[None,None,:].expand(-1,xyz_coord.shape[1],-1)
        input_all = torch.cat([xyz_coord, code_repeat], -1)  # 1 x N x (C+3)
        sdf_pred = decoder(input_all)
        sdf_im = sdf_pred.reshape([resol[0], resol[1]])
        volume[z_ind] = sdf_im
    return volume

def extract_mesh_unet(unet, dec, z_latent, grid_trans, resol, ranges, save_name, level=0.0):
    # resol and ranges order: x y z
    z_conv = unet(z_latent)
    print(z_conv.shape)
    z_flat = z_conv.view([z_conv.shape[0], z_conv.shape[1], -1])
    sdf_volume = compute_sdf_3D_unet(z_flat, grid_trans, dec, resol, ranges)

    convert_sdf_samples_to_ply(sdf_volume, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], save_name, level=level)
    # return sdf_volume

def extract_mesh_generator(gen, dec, z_latent, resol, ranges, save_name, level=0.0):
    # resol and ranges order: x y z
    with torch.no_grad():
        codebook = gen(z_latent.detach())
        try: s=codebook.shape
        except: codebook = codebook[0]
        sdf_volume = compute_sdf_3D_gen(codebook, dec, resol, ranges)

        convert_sdf_samples_to_ply(sdf_volume, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], save_name, level=level)
        return sdf_volume

def extract_mesh_deepsdf(decoder, z_latent, resol, ranges, save_name, level=0.0, out_mesh=False):
    with torch.no_grad():
        sdf_volume = compute_sdf_3D_deepsdf(decoder, z_latent, resol, ranges)
        outs = convert_sdf_samples_to_ply(sdf_volume, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], save_name, level=level, out_meshdata=out_mesh)
        if out_mesh:
            return sdf_volume, outs
        else: return sdf_volume


def extract_mesh_generator_multi(gen, dec, z_latent, resol, ranges, save_name, level=0.0):
    # resol and ranges order: x y z
    with torch.no_grad():
        num_z = z_latent.shape[0]
        assert(num_z>1)
        codebook = gen(z_latent.detach())
        try: s=codebook.shape
        except: codebook = codebook[0]
        sdf_volumes = []
        for i in range(num_z):
            codebook1 = codebook[i:i+1]
            sdf_volumes.append(compute_sdf_3D_gen(codebook1, dec, resol, ranges))
        sdf_volume_stack = torch.stack(sdf_volumes)
        sdf_volume = sdf_volume_stack.min(dim=0)[0]

        convert_sdf_samples_to_ply(sdf_volume, [0.,0.,0.], (ranges[0][1]-ranges[0][0]) / resol[0], save_name, level=level)
        return sdf_volume


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
    out_meshdata=False
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 2]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 0]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

    print('marching cubes took: {}'.format(time.time() - start_time))
    if out_meshdata: return ply_data


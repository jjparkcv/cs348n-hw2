from shapes import *
import torch.optim as optim
from torch.utils.data import Dataset
from datetime import date
import os
import argparse
import sys
from torch.autograd import grad
import random
import argparse
import os
import random
import sys
from datetime import date

import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import Dataset

from shapes import *
from extract_mesh import *

import torch
torch.manual_seed(1)
import random
random.seed(1)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Attention')
    parser.add_argument('--l2_regul', default=1e-3, type=float, help='learning rate in training [default: 2e-4]')
    parser.add_argument('--trunc', default=0.1, type=float, help='learning rate in training [default: 2e-4]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--name', type=str, default='model', help='specify gpu device [default: 0]')
    parser.add_argument('--load_model', type=str, default=None, help='specify gpu device [default: 0]')
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# python -i shape_completion_deepsdf.py --load_model weights.pth

data_dir = 'point_test/'
class PointsNormalDataset(Dataset):
    def __init__(self, data_direc, num_sample,start_shape_ind=0, num_shapes=None, pre_load=True, specify_ind=None,
                 mask_out=None, num_mask=1, depthmap=False):
        self.data_dir = data_direc
        self.num_sample = num_sample
        self.mask_out = mask_out
        self.num_mask = num_mask
        self.depthmap = depthmap
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]
        self.files.sort()
        if specify_ind is not None:
            self.files = self.files[specify_ind:specify_ind+1]
        elif num_shapes is not None:
            self.files = self.files[start_shape_ind:start_shape_ind+num_shapes]
        self.pre_loaded = []
        self.pre_load = pre_load
        if self.pre_load:
            self.pre_load_files()
        print('loaded '+str(len(self)) + ' scenes.')
    def __len__(self):
        return len(self.pre_loaded)
    def masking_cube(self, points):
        i = torch.randperm(points.shape[0])[0]
        return Shapes.mask_out_pts_cube(points, points[i, :3], self.mask_out)

    def pre_load_files(self):
        if len(self.pre_loaded) > 0: assert False
        for i,fname in enumerate(self.files):
            print(i)
            loaded = torch.from_numpy(np.load(fname)).float()
            loaded = loaded[~torch.isnan(loaded).sum(1).bool(), :]
            if self.depthmap:
                loaded = Shapes.get_depthmap_no_occlusion(loaded, camera= (0.5774,0.5774,-0.5774), lookat=(0.0,0.0,0.))
            self.pre_loaded.append(loaded)
    def __getitem__(self, idx):
        rand_select = torch.randperm(self.pre_loaded[idx].shape[0])
        selected = self.pre_loaded[idx][rand_select[:self.num_sample], :]
        pts = selected[:, :3]
        norms = selected[:, 3:]
        return pts, norms, idx


truncation = args.trunc
l2_regul = args.l2_regul
input_range = [[-0.5,0.5]]*3

#############################################################################
# Complete this function
def train_epoch(epoch_curr, train_loader, z, opt_z, opt_net=None):
    loss_sum = 0.

    for batch_i, data in enumerate(train_loader):
        surf_samples = data[0].cuda()  # B x N x 3
        num_xyz = surf_samples.shape[1]
        norm_in = data[1].cuda()  # B x N x 3
        idx = data[2].cuda()

        rand_pts = surf_samples + torch.randn_like(surf_samples) * truncation
        pts_all = torch.cat([surf_samples, rand_pts], 1)  # B x 2N x 3
        pts_all.requires_grad = True

        if opt_net is not None: opt_net.zero_grad()
        opt_z.zero_grad()
        N = surf_samples.shape[1]
        z_batch = torch.index_select(z, 0, idx)  # B x C
        ##############################################################
        z_repeat = z_batch.unsqueeze(1).expand(-1, 2*N, -1)  # B x 2N x C

        pts_input = ...  # B x 2N x (C+3)
        sdf_pred = ...  # B x 2N x 1
        # B x 2N x 3
        normal_raw = grad(...

        # sdf_pred: B x 2N x 1, normal_raw: B x 2N x 3
        loss_sdf = ...
        normal_pred = ...
        loss_normal = ...
        loss_eikonal = ...
        z_regul = ...
        ############################################################
        loss_empty = torch.exp(-100 * sdf_pred[:, num_xyz:].abs()).mean()
        loss_all = z_regul

        loss_all += 1.*loss_sdf + 0.2*loss_normal + 0.2*loss_eikonal + 0.25*loss_empty
        loss_all.backward()

        if opt_net is not None: opt_net.step()
        opt_z.step()

        loss_sum += loss_all.detach().item()
        if e % 100 == 0:
            print('sdf: ' + str(loss_sdf.cpu().detach().numpy()))
            print('normal: ' + str(loss_normal.cpu().detach().numpy()))
            print('eik: ' + str(loss_eikonal.cpu().detach().numpy()))

    return loss_sum / (batch_i+1)

##################################################################################
def save_checkpoint(path, epoch, dec, optimizer_z, optimizer_net, zs, argss, loss_curve):
    torch.save({
        'epoch': epoch,
        'optimizer_z': optimizer_z.state_dict(),
        'optimizer_net': optimizer_net.state_dict(),
        'decoder': dec,
        'zs': zs,
        'args': argss,
        'loss_curve': loss_curve
    }, path)

def load_checkpoint(path):
    l = torch.load(path)
    dec = l['decoder'].cuda()
    zz = l['zs'].cuda()
    return dec, zz

import time
decoder, z_loaded = load_checkpoint(args.load_model)
os.system('mkdir -p output/')

for ii in range(2):
    z_new = torch.randn(1,z_loaded.shape[1]).cuda() * 0.01
    z_new.requires_grad = True
    optimizer_z = optim.Adam([z_new], lr=1e-1)
    
    dataset_inf = PointsNormalDataset(data_dir, 2048, specify_ind=ii, depthmap=True)
    print(dataset_inf.files[0])
    writePly(os.path.join('output', str(ii)+'-depth.ply'), dataset_inf.pre_loaded[0])

    train_loader = torch.utils.data.DataLoader(
        dataset_inf, batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )
    
    for e in range(1000):
        l = train_epoch(e, train_loader, z_new, optimizer_z, None)
        if e % 100 == 0:
            print(z_new.std())
    extract_mesh_deepsdf(decoder, z_new[0], [256,256,256],
                               input_range, os.path.join('output', str(ii) + '.ply'))
    
    
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
from models.code_conv3D import *

# Run this command for training
# python -i single_scene.py --epoch 2000 --name deepsdf --gpu 0 --trunc 0.1 --lr 2e-4

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Attention')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--num_scenes', default=300, type=int)
    parser.add_argument('--num_shapes', default=6, type=int)
    parser.add_argument('--num_latent', default=4, type=int)
    parser.add_argument('--dim_latent', default=256, type=int)
    parser.add_argument('--dim_inner', default=512, type=int)
    parser.add_argument('--dim_decoder', default=32, type=int)
    parser.add_argument('--normal', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_z', default=6e-4, type=float)
    parser.add_argument('--l2_regul', default=1e-4, type=float)
    parser.add_argument('--trunc', default=0.1, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--num_point', type=int, default=2048)
    parser.add_argument('-t', '--test', dest='testing', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

training = not args.testing

experiments_dir = 'training/'

def create_date_folder():  # models, loss graphs, and visualizations will be saved in this directory
    os.system('mkdir -p training/')
    return 'training'

data_dir = 'data/03001627'

class SDFDataset(Dataset):
    def __init__(self, data_direc, num_sample, num_shapes=None, truncation=0.1):
        self.data_dir = data_direc
        self.num_sample = num_sample
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]
        self.files.sort()
        self.num_shapes = num_shapes
        if num_shapes is not None: self.files = self.files[:num_shapes]
        self.pre_loaded = []
        self.truncation = truncation
        self.pre_load_files()
        print('loaded ' + str(len(self)) + ' scenes.')

    def __len__(self):
        return len(self.pre_loaded)

    def pre_load_files(self):
        if len(self.pre_loaded) > 0: assert False
        for i, fname in enumerate(self.files):
            if i % 300 == 0:
                print(i)
            try:
                loaded = np.load(fname)
                loaded_neg = torch.from_numpy(loaded['neg']).float()
                loaded_pos = torch.from_numpy(loaded['pos']).float()
                loaded_neg = loaded_neg[~torch.isnan(loaded_neg).sum(1).bool(), :]
                loaded_pos = loaded_pos[~torch.isnan(loaded_pos).sum(1).bool(), :]
                assert (torch.all(~torch.isnan(loaded_neg)))
                assert (torch.all(~torch.isnan(loaded_pos)))
                self.pre_loaded.append([loaded_neg, loaded_pos])
            except:
                print(fname)
                os.system('rm ' + fname)

    @staticmethod
    def sample_pts(pts, num_sample):
        rand_select = torch.randperm(pts.shape[0])
        return pts[rand_select[:num_sample], :]

    def __getitem__(self, idx):
        neg_select = self.sample_pts(self.pre_loaded[idx][0], self.num_sample // 2)
        pos_select = self.sample_pts(self.pre_loaded[idx][1], self.num_sample // 2)
        pts_select = torch.cat([neg_select, pos_select], 0)
        return pts_select, idx


epoch_start = 0
batch_size = args.batch_size

truncation = args.trunc
samp_per_scene = args.num_point
num_scenes = args.num_scenes if training else 10
epoch = args.epoch
l2_regul = args.l2_regul
lr = args.lr
lr_z = args.lr_z

input_range = [[-0.5, 0.5]] * 3

latent_size = args.dim_latent
num_latent = args.num_latent

decoder = nn.DataParallel(
    DecoderSimple(0, [512, 512, 512, 512, 512, 512, 512], 3, weight_norm=True, latent_in=[4],
                  swish=True, swish_beta=15., norm_layers=list(range(10)))).cuda()

optimizer_net = optim.Adam(decoder.parameters(), lr=lr)

dataset = SDFDataset(data_dir, 10000, num_shapes=1)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True
)


def visualize_slice(dec, zslice=0., save_name=None, name=''):
    im_res = 360
    xy_coord = Shape.grid_coords(im_res).unsqueeze(0).cuda() - 0.5
    xyz_coord = torch.cat([xy_coord, torch.ones(1, xy_coord.shape[1], 1).cuda() * zslice], 2)  # 1 x N x 3
    sdf_pred = dec(xyz_coord.cuda())

    if save_name is not None:
        sdf_im = sdf_pred.detach().reshape([im_res, im_res])
        Canvas.save_image(sdf_im.cpu().numpy(), os.path.join(save_name, name + 'inf.png'))
        re = None
        return re


import matplotlib.pyplot as plt

loss_curve = []
loss_epoch = []

def plot_loss(dir):
    if len(loss_epoch) < 5: return
    plt.clf()
    plt.plot(loss_epoch[5:], loss_curve[5:])
    plt.savefig(os.path.join(dir, 'loss_curve_test.png'), bbox_inches='tight')


#############################################################################
# This function you need to complete
def train_epoch(train_loader, opt_net):
    for batch_i, data in enumerate(train_loader):
        xyz_pts = data[0][..., :3].cuda()  # B x N x 3
        sdf_gt = data[0][..., 3:4].cuda()  # B x N x 1

        opt_net.zero_grad()
        ################################################
        # implement loss = |F(x)-SDF(x)|
        sdf_pred = ...
        sdf_loss = ...      
        ################################################
        sdf_loss.backward()
        opt_net.step()
    return sdf_loss.detach().item()


import time

if __name__ == "__main__":
    save_point = os.path.join(create_date_folder(), args.name)

    os.system('mkdir -p ' + save_point)
    vis = True

    checkpoint_every = 300
    for e in range(epoch_start, epoch):
        l = train_epoch(train_loader, optimizer_net)
        loss_curve.append(l)
        loss_epoch.append(e)
        plot_loss(save_point)
        if e%20 == 0: print(l)
        if (e == 2) or (e != 0 and e % 5 == 0):
            if vis:
                with torch.no_grad():
                    visualize_slice(decoder, save_name=save_point, zslice=0.1)

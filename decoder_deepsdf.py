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

## Run this command for training
# python -i decoder_deepsdf.py --epoch 2000 --batch_size 16 --num_scenes 100 --name deepsdf --gpu 0 --trunc 0.1 --dim_latent 256 --lr 1e-3 --lr_z 2e-3

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

def create_date_folder():
    os.system('mkdir -p training/')
    return 'training'


def load_model(mod):
    path = os.path.join(experiments_dir, mod)
    loaded = torch.load(path)
    dec_loaded = loaded['decoder'].cuda()
    z_loaded = loaded['zs'].cuda()
    epoch_loaded = loaded['epoch'] + 1
    opt_z_dict = loaded['optimizer_z']
    opt_net_dict = loaded['optimizer_net']
    return dec_loaded, z_loaded, epoch_loaded, opt_z_dict, opt_net_dict

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
    DecoderSimple(latent_size, [512, 512, 512, 512, 512, 512, 512], 3, weight_norm=True, latent_in=[4],
                  swish=True, swish_beta=15., norm_layers=list(range(10)))).cuda()

z = torch.zeros(num_scenes, latent_size).normal_(mean=0.0, std=0.001).cuda()  # B x N x L
z.requires_grad = True

optimizer_net = optim.Adam(decoder.parameters(), lr=lr)
optimizer_z = optim.Adam([z], lr=lr_z)


def warmup_lr(start_lr, end_lr, num_epoch, curr_epoch, opt):
    if curr_epoch > num_epoch: return
    step_size = (end_lr - start_lr) / num_epoch
    _lr = start_lr + step_size * curr_epoch
    for g in opt.param_groups: g['lr'] = _lr


dataset = SDFDataset(data_dir, samp_per_scene, num_shapes=num_scenes)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True
)


def visualize_slice(dec, z_code, zslice=0., save_name=None, name=''):
    im_res = 360
    xy_coord = Shape.grid_coords(im_res).unsqueeze(0).cuda() - 0.5
    xyz_coord = torch.cat([xy_coord, torch.ones(1, xy_coord.shape[1], 1).cuda() * zslice], 2)  # 1 x N x 3
    z_sel = z_code.unsqueeze(0).expand(-1, xyz_coord.shape[1], -1)  # 1 x N x C
    input_all = torch.cat([xyz_coord.cuda(), z_sel], -1)  # 1 x N x (3+C)
    sdf_pred = dec(input_all)

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
# Complete this function
def train_epoch(epoch_curr, train_loader, z_in, opt_z, opt_net=None):
    loss_sum = 0.
    print_every = 10

    for batch_i, data in enumerate(train_loader):
        xyz_pts = data[0][..., :3].cuda()  # B x N x 3
        sdf_gt = data[0][..., 3:4].cuda()  # B x N x 1
        num_xyz = xyz_pts.shape[1]
        idx = data[1].cuda()

        if opt_net is not None: opt_net.zero_grad()
        opt_z.zero_grad()
        ##################################################
        z_batch = torch.index_select(z_in, 0, idx)  # B x C
        ...
        sdf_loss = ...
        z_regul = ...
        ##################################################
        loss_all = z_regul + sdf_loss
        loss_all.backward()

        if opt_net is not None: opt_net.step()
        opt_z.step()

        loss_sum += loss_all.detach().item()

        if (batch_i + 5) % print_every == 0:
            log = 'Train Epoch: {} [{}/{}]  name:{}  z_std: {:.5f} SDF: {:.5f}'.format(
                epoch_curr, batch_i * batch_size, len(train_loader.dataset), args.name, z_in.std(),
                sdf_loss.detach().item())
            print(log)
    return loss_sum / (batch_i + 1)


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

if training:
    save_point = os.path.join(create_date_folder(), args.name)
    if args.load_model is not None: save_point = os.path.join(experiments_dir, os.path.dirname(args.load_model))

    os.system('mkdir -p ' + save_point)
    writePly(os.path.join(save_point, 'first.ply'), dataset.pre_loaded[0][1])

    vis = True

    checkpoint_every = 300
    for e in range(epoch_start, epoch):
        t1 = time.time()
        l = train_epoch(e, train_loader, z, optimizer_z, optimizer_net)
        print(time.time() - t1)
        loss_curve.append(l);
        loss_epoch.append(e)
        plot_loss(save_point)
        if (e == 2) or (e != 0 and e % 5 == 0):
            if vis:
                with torch.no_grad():
                    visualize_slice(decoder, z[0].unsqueeze(0), save_name=save_point, zslice=0.1)

        if e != 0 and e % checkpoint_every == 0:
            save_checkpoint(
                os.path.join(save_point, str(e) + '.pth'),
                e, decoder, optimizer_z, optimizer_net, z, args, loss_curve
            )
            extract_mesh_deepsdf(decoder, z[0], [256, 256, 256],
                                 input_range, os.path.join(save_point, 'scene0.ply'))

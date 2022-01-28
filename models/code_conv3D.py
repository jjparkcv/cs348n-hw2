import torch
import torch.nn.functional as F
import torch.nn as nn

#extract_mesh_unet(a['encoder'],a['decoder'],a['zs'][6:7],grid_trans,[320,320,160], [(0.,k) for k in input_range],os.path.join(save_point, 'scene0.ply'))
class GridTransform3D():
    def __init__(self, input_range, resolution, overlap=True, normalize=True, overlap_ratio=0.5):
        # input_range is a length 3 list consists of maximum values for each dimension
        # Each dimension is assumed to have range 0 ~ input_range_k where k is either x, y, or z
        # resolution is similarly a Long tensor of length 3 that consists of resolution for each dimension, orde: xyz
        # self.vox_length has to be same for all dimensions meaning that voxel has to be a cube
        self.input_range = input_range  # not assuming cube
        self.resolution = resolution
        self.overlap = overlap
        if self.overlap: self.overlap_ratio = overlap_ratio
        self.normalize = normalize
        self.vox_length = input_range[0] / float(self.resolution[0])

    def consistency_index(self, coords):
        overlap = self.overlap
        self.overlap = False   # Let's explore setting this to False and then True
        xy_ind1, rel_coord1 = self.compute_index(coords)
        self.overlap = True
        xy_ind2, rel_coord2 = self.compute_index(coords)
        self.overlap = overlap
        return xy_ind1, rel_coord1, xy_ind2, rel_coord2

    def consistency_index2(self, coords):
        overlap = self.overlap
        self.overlap = False   # Let's explore setting this to False and then True
        xy_ind1, rel_coord1 = self.compute_index(coords)
        self.overlap = True
        xy_ind2, rel_coord2, shiftability = self.compute_index(coords, return_shifts=True)
        self.overlap = overlap
        return xy_ind1, rel_coord1, xy_ind2, rel_coord2


    @staticmethod
    def clamp_dim(tensor, min_clamp, max_clamp, assert_error=False):
        # tensor: B x N x 3
        for i in range(tensor.shape[2]):
            if assert_error:
                assert(tensor[:,:,i].max() <= float(max_clamp[i]))
                assert(tensor[:,:,i].min() >= float(min_clamp[i]))
            else:
                tensor[:,:,i] = tensor[:,:,i].clamp(min=float(min_clamp[i]), max=float(max_clamp[i]))
        if not assert_error:
            return tensor

    def compute_index(self, coords, return_shifts=False):
        # coords: B x N x 3
        lastdim = len(coords.shape) - 1
        coords_pos = coords.detach()  # assume all positive inputs
        coord_ind = torch.div(coords_pos, self.vox_length).long() #.clamp(max=self.resolution - 1, min=0).long()
        self.clamp_dim(coord_ind, [0,0,0], self.resolution-1)
        # coordinate index for each dimension
        coord_mod = coords_pos - coord_ind * self.vox_length  # remainder of the coordinate # 0~vox_length
        if self.overlap:
            overlap_ratio = 0.5 if not hasattr(self, 'overlap_ratio') else self.overlap_ratio
            shiftability = torch.sign(coord_mod-self.vox_length*overlap_ratio) / 2. + \
                torch.sign(coord_mod+self.vox_length*(overlap_ratio-1)) / 2.
            # shiftability = torch.sign(coord_mod - self.vox_length * 0.5)  # -1 or 1
            rand = torch.randint(0, 2, shiftability.shape)
            if coords.is_cuda: rand = rand.cuda()
            rand_shift = shiftability * rand
            coord_ind = self.clamp_dim((coord_ind + rand_shift.long()), [0,0,0], self.resolution - 1)

        index_center = coord_ind * self.vox_length + self.vox_length * 0.5
        relative_coord = coords_pos - index_center  # B x N x 2 or B x 2
        if self.normalize: relative_coord /= self.vox_length * 0.5

        x_ind = torch.index_select(coord_ind, lastdim, torch.tensor([0]).long().cuda())
        y_ind = torch.index_select(coord_ind, lastdim, torch.tensor([1]).long().cuda())
        z_ind = torch.index_select(coord_ind, lastdim, torch.tensor([2]).long().cuda())
        xyz_ind = z_ind * (self.resolution[0]*self.resolution[1]) + y_ind * self.resolution[0] + x_ind  # B x N x 1

        if self.overlap and return_shifts:
            return xyz_ind.squeeze(-1), relative_coord, shiftability
        else: return xyz_ind.squeeze(-1), relative_coord


# if __name__ == "__main__":
#     g3 = GridTransform3D([1.0,1.0,0.5], torch.tensor([10,10,5]).long(), overlap=False)
#     g3_over = GridTransform3D([1.0,1.0,0.5], torch.tensor([10,10,5]).long(), overlap=True)
#     t = torch.rand(1,4,3).cuda(); t[:,:,2] *= 0.5
#     xyz, rel = g3.compute_index(t)
#     xyz2, rel2 = g3_over.compute_index(t)


def conv3(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    padder = None
    to_pad = kernel_size // 2
    if pad == 'reflection':
        padder = nn.ReflectionPad3d(to_pad)
        to_pad = 0

    convolver = nn.Conv3d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


class UpConv3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, transpose=False, upsamp='nearest', weight_norm=False, skip=True):
        super(UpConv3D, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if skip: mid_dim = in_channels//2
        else: mid_dim = in_channels
        if upsamp == 'nearest': self.up = nn.Upsample(scale_factor=2, mode=upsamp)
        else: self.up = nn.Upsample(scale_factor=2, mode=upsamp, align_corners=True)
        self.conv_up = nn.Conv3d(in_channels, mid_dim, kernel_size=3, padding=1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, mid_dim, kernel_size=2, stride=2)
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
            self.conv_up = nn.utils.weight_norm(self.conv_up)

    def forward(self, x1, skip=None):
        x = F.leaky_relu(self.conv_up(self.up(x1)), 0.2)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)



class DownConv3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, weight_norm=False):
        super(DownConv3D, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
            self.conv2 = nn.utils.weight_norm(self.conv2)

    def forward(self, x):
        return self.conv2(F.leaky_relu(self.conv(self.maxpool(x)), 0.2))


class UNet3D(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, weight_norm=False):
        super(UNet3D, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=5, padding=2)
        self.down1 = DownConv3D(mid_dim, mid_dim*2, weight_norm=weight_norm)
        self.down2 = DownConv3D(mid_dim*2, mid_dim*4, weight_norm=weight_norm)
        self.up1 = UpConv3D(mid_dim*4, mid_dim*2, weight_norm=weight_norm)
        self.up2 = UpConv3D(mid_dim*2, mid_dim, skip=False, weight_norm=weight_norm)
        self.conv3 = nn.Conv3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, padding=1)
        self.weight_norm = weight_norm
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv3 = nn.utils.weight_norm(self.conv3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        x1 = self.relu(self.conv1(code_im))
        x2 = self.relu(self.down1(x1))
        x3 = self.relu(self.down2(x2))

        x4 = self.relu(self.up1(x3, x2))
        x5 = self.relu(self.up2(x4))
        x = self.relu(self.conv3(x5))
        x = self.conv4(x)
        return x


class UNet3DFull(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, weight_norm=False):
        super(UNet3DFull, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=5, padding=2)
        self.down1 = DownConv3D(mid_dim, mid_dim*2, weight_norm=weight_norm)
        self.down2 = DownConv3D(mid_dim*2, mid_dim*4, weight_norm=weight_norm)
        self.up1 = UpConv3D(mid_dim*4, mid_dim*2, weight_norm=weight_norm)
        self.up2 = UpConv3D(mid_dim*2, mid_dim, weight_norm=weight_norm)
        self.conv3 = nn.Conv3d(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, padding=1)
        self.weight_norm = weight_norm
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv3 = nn.utils.weight_norm(self.conv3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        x1 = self.relu(self.conv1(code_im))
        x2 = self.relu(self.down1(x1))
        x3 = self.relu(self.down2(x2))

        x4 = self.relu(self.up1(x3, x2))
        x5 = self.relu(self.up2(x4, x1))
        x = self.relu(self.conv3(x5))
        x = self.conv4(x)
        return x


class UpConv3DSimple(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, skip=False, stride=1, skip_channels=None,
                 instance_norm=True, weight_norm=False, swish=None, padding_mode='zeros', upsamp_mode='nearest'):
        super(UpConv3DSimple, self).__init__()
        self.upsample = 2
        mid_channel = out_channels
        self.swish = swish
        self.relu = nn.LeakyReLU(0.2)
        self.skip = skip
        self.instance_norm = instance_norm; self.weight_norm = weight_norm
        if skip: mid_channel = in_channels // 2
        # reflection_padding = kernel_size // 2
        self.padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(in_channels, mid_channel, kernel_size, stride)
        self.in2 = torch.nn.InstanceNorm3d(mid_channel, affine=True, track_running_stats=True)
        self.upsamp_mode = upsamp_mode
        if weight_norm: self.conv1 = nn.utils.weight_norm(self.conv1)
        if skip:
            if skip_channels is None:
                self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
            else:
                self.conv2 = torch.nn.Conv3d(skip_channels+mid_channel, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
            if weight_norm: self.conv2 = nn.utils.weight_norm(self.conv2)

    def forward(self, x, skip=None):
        relu = self.relu
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
        x_in = x
        mode = 'nearest'
        if hasattr(self, 'upsamp_mode'): mode = self.upsamp_mode
        if self.upsample:
            x_in = F.interpolate(x_in, mode=mode, scale_factor=self.upsample)
        out = F.pad(x_in, [self.padding]*6, 'replicate')  # you don't need to use this, just use convolution with padding
        if self.instance_norm: out = relu(self.in2(self.conv1(out)))
        else: out = relu(self.conv1(out))
        if self.skip:
            cat = torch.cat([out, skip], 1)
            out = self.conv2(cat)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, weight_norm=False, swish=None, padding_mode='zeros'):
        super(DoubleConv, self).__init__()
        self.padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding=self.padding)
        self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=self.padding)
        self.weight_norm = weight_norm
        self.swish = swish
        self.padding_mode = padding_mode
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)
    def forward(self, x):
        x = self.conv1(x)
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
            x = relu(x)
        x = self.conv2(x)
        return x


class ConvVAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, weight_norm=False, swish=None, padding_mode='zeros'):
        super(ConvVAE, self).__init__()
        # self.doubleConv = DoubleConv(in_channels, out_channels*2, kernel_size, stride=stride, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
        self.doubleConv = nn.Conv3d(in_channels, out_channels*2, kernel_size, stride=stride, padding_mode=padding_mode, padding=kernel_size//2)
        if weight_norm: self.doubleConv = nn.utils.weight_norm(self.doubleConv)
        self.out_chan = out_channels

    def remove_weight_norm(self):
        self.doubleConv.remove_weight_norm()

    def forward(self, x):
        x = self.doubleConv(x)
        mu = x[:,:self.out_chan]
        logvar = x[:,self.out_chan:]
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        feature = mu + std * eps
        return mu, feature, logvar


class UpConv3DNew(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, skip=False, stride=1, skip_channels=None,
                 instance_norm=True, weight_norm=False, swish=None, padding_mode='zeros', upsamp_mode='nearest'):
        super(UpConv3DNew, self).__init__()
        self.upsample = 2
        mid_channel = out_channels
        self.swish = swish
        self.relu = nn.LeakyReLU(0.2)
        self.skip = skip
        self.instance_norm = instance_norm; self.weight_norm = weight_norm
        if skip: mid_channel = in_channels // 2
        # reflection_padding = kernel_size // 2
        self.padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(in_channels, mid_channel, kernel_size, stride, padding=self.padding)
        self.in2 = torch.nn.InstanceNorm3d(mid_channel, affine=True, track_running_stats=True)
        self.upsamp_mode = upsamp_mode
        if weight_norm: self.conv1 = nn.utils.weight_norm(self.conv1)
        if skip:
            if skip_channels is None:
                self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
            else:
                self.conv2 = torch.nn.Conv3d(skip_channels+mid_channel, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
        else:
            self.conv2 = torch.nn.Conv3d(mid_channel, mid_channel, kernel_size, stride, padding=self.padding)
        if weight_norm: self.conv2 = nn.utils.weight_norm(self.conv2)
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)
    def forward(self, x, skip=None):
        relu = self.relu
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
        x_in = x
        mode = 'nearest'
        if hasattr(self, 'upsamp_mode'): mode = self.upsamp_mode
        if self.upsample:
            x_in = F.interpolate(x_in, mode=mode, scale_factor=self.upsample)
        if self.instance_norm: out = relu(self.in2(self.conv1(x_in)))
        else: out = relu(self.conv1(x_in))

        if self.skip:
            cat = torch.cat([out, skip], 1)
            out = self.conv2(cat)
        else:
            out = self.conv2(out)
        return out


class UpConv3DNew2(nn.Module):
    """Conv then Upscale then Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, skip=False, stride=1, skip_channels=None,
                 instance_norm=True, weight_norm=False, swish=None, padding_mode='zeros', upsamp_mode='nearest'):
        super(UpConv3DNew2, self).__init__()
        self.upsample = 2
        mid_channel = out_channels
        self.swish = swish
        self.relu = nn.LeakyReLU(0.2)
        self.skip = skip
        if skip: mid_channel = in_channels // 2
        # reflection_padding = kernel_size // 2
        self.padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(in_channels, mid_channel, kernel_size, stride, padding=self.padding)
        self.conv1_1 = torch.nn.Conv3d(mid_channel, mid_channel, kernel_size, stride, padding=self.padding)
        self.upsamp_mode = upsamp_mode
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv1_1 = nn.utils.weight_norm(self.conv1_1)
        if skip:
            if skip_channels is None:
                self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
            else:
                self.conv2 = torch.nn.Conv3d(skip_channels+mid_channel, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
        else:
            self.conv2 = torch.nn.Conv3d(mid_channel, mid_channel, kernel_size, stride, padding=self.padding)
        if weight_norm: self.conv2 = nn.utils.weight_norm(self.conv2)
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)
    def forward(self, x, skip=None):
        relu = self.relu
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
        x_in = x
        mode = 'nearest'
        if hasattr(self, 'upsamp_mode'): mode = self.upsamp_mode
        out = self.conv1(x_in)
        if self.upsample:
            out = F.interpolate(out, mode=mode, scale_factor=self.upsample)
        out = relu(out)
        out = relu(self.conv1_1(out))

        if self.skip:
            cat = torch.cat([out, skip], 1)
            out = self.conv2(cat)
        else:
            out = self.conv2(out)
        return out


class ResidualBlock3D(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, weight_norm=True, swish=None, padding_mode='zeros'):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.relu = nn.LeakyReLU(0.2)
        self.weight_norm = weight_norm
        self.swish = swish
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)

    def forward(self, x):
        relu = self.relu
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
        residual = x
        out = relu(self.conv1(x))
        out = self.conv2(out)
        out = relu(out + residual)
        return out


class ResidualBlockConcat(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, mid_channels, weight_norm=True, swish=None):
        super(ResidualBlockConcat, self).__init__()
        self.conv1 = nn.Conv3d(channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(mid_channels+channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.weight_norm = weight_norm
        self.swish = swish
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)

    def forward(self, x):
        relu = self.relu
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish * t) * t
        out = relu(self.conv1(x))
        out = relu(self.conv2(torch.cat([out, x], 1)))
        return out


class UNet3D3_Res(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros'):
        super(UNet3D3_Res, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode

        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*8, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode) #try changing to mid_dim*8

        self.res_blocks = nn.ModuleList([ResidualBlock3D(mid_dim*8,  weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
                                         for i in range(num_res)])

        # self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*8, swish=swish, padding_mode=padding_mode)
        self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
        self.up2 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.down3 = nn.utils.weight_norm(self.down3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im)) #mid
        x2 = relu(self.down1(x1)) #mid*2
        x3 = relu(self.down2(x2)) #mid*4
        x4 = relu(self.down3(x3))

        for res in self.res_blocks:
            x4 = res(x4)

        x5 = relu(self.up0(x4, x3))
        x6 = relu(self.up1(x5, x2))
        x7 = relu(self.up2(x6, x1))

        x = self.conv4(x7)

        return x



class Net3D3_Default(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=3, num_res=1, padding_mode='zeros'):
        super(Net3D3_Default, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode

        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=3, padding=kernel_size//2, padding_mode=padding_mode)
        self.res_blocks = nn.ModuleList([ResidualBlock3D(mid_dim,  weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
                                         for i in range(num_res)])
        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=1, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x = relu(self.conv1(code_im)) #mid
        for res in self.res_blocks:
            x = res(x)
        x = self.conv4(x)

        return x


class UpFC3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip=False, stride=1, skip_channels=None,
                 instance_norm=True, weight_norm=False, swish=None, padding_mode='zeros', upsamp_mode='nearest'):
        super(UpFC3D, self).__init__()
        mid_channel = out_channels
        self.swish = swish
        self.relu = nn.LeakyReLU(0.2)
        self.skip = skip
        if skip: mid_channel = in_channels // 2
        self.padding = kernel_size // 2
        self.conv1 = torch.nn.Conv3d(in_channels, mid_channel, kernel_size, stride, padding=self.padding)
        self.conv1_1 = torch.nn.Conv3d(mid_channel, mid_channel, kernel_size, stride, padding=self.padding)
        self.upsamp_mode = upsamp_mode
        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv1_1 = nn.utils.weight_norm(self.conv1_1)
        if skip:
            if skip_channels is None:
                self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
            else:
                self.conv2 = torch.nn.Conv3d(skip_channels+mid_channel, out_channels, kernel_size, stride, padding=self.padding, padding_mode=padding_mode)
        else:
            self.conv2 = torch.nn.Conv3d(mid_channel, mid_channel, kernel_size, stride, padding=self.padding)
        if weight_norm: self.conv2 = nn.utils.weight_norm(self.conv2)



class UNetFC(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, num_res=1):
        super(UNetFC, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.max_pool = nn.MaxPool3d((2, 2, 2))

        self.conv1 = DoubleConv(in_dim, mid_dim*2, kernel_size=1, swish=swish, weight_norm=weight_norm)
        self.down1 = DoubleConv(mid_dim*2, mid_dim*4, kernel_size=1, swish=swish, weight_norm=weight_norm)
        self.down2 = DoubleConv(mid_dim*4, mid_dim*8, kernel_size=1, swish=swish, weight_norm=weight_norm)


class UNetMaxPool(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, num_res=1, upsamp_mode='trilinear'):
        super(UNetMaxPool, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.upsamp_mode = upsamp_mode
        self.max_pool = nn.MaxPool3d((2, 2, 2))

        self.conv1 = DoubleConv(in_dim, mid_dim*2, kernel_size=1, swish=swish, weight_norm=weight_norm)
        self.down1 = DoubleConv(mid_dim*2, mid_dim*4, kernel_size=1, swish=swish, weight_norm=weight_norm)
        self.down2 = DoubleConv(mid_dim*4, mid_dim*8, kernel_size=1, swish=swish, weight_norm=weight_norm)
        # Try resnet blocks on this middle layers
        self.mid_blocks = nn.ModuleList([DoubleConv(mid_dim * 8, mid_dim * 8, kernel_size=3, swish=swish, weight_norm=weight_norm) for _ in range(num_res)])
        self.up0 = UpConv3DNew2(mid_dim * 8, mid_dim * 4, kernel_size=1, skip=True, weight_norm=weight_norm, swish=swish, upsamp_mode=upsamp_mode)
        self.up1 = UpConv3DNew2(mid_dim * 4, mid_dim * 2, kernel_size=1, skip=True, weight_norm=weight_norm, swish=swish, upsamp_mode=upsamp_mode)
        self.up2 = UpConv3DNew2(mid_dim * 2, mid_dim * 1, kernel_size=1, skip=False, weight_norm=weight_norm, swish=swish, upsamp_mode=upsamp_mode)
        self.conv2 = nn.Conv3d(mid_dim, out_dim, kernel_size=1, stride=1, padding=1)

        if weight_norm:
            self.conv2 = nn.utils.weight_norm(self.conv2)

    def forward(self, code_im, skip_noise=None):
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im))
        x2 = self.max_pool(relu(self.down1(x1)))
        x3 = self.max_pool(relu(self.down2(x2)))

        x_mid = x3
        # for res in self.mid_blocks:
        #     x_mid = relu(res(x_mid))ire


        if skip_noise is None:
            s2=x2; s1=x1;
        else:
            s1 = F.dropout(x1, skip_noise[0]); s2 = F.dropout(x2, skip_noise[1])

        # x5 = relu(self.up0(x_mid, s2))
        # x6 = relu(self.up1(x5, s1))
        x5 = relu(self.up0(x_mid, torch.zeros_like(s2)))
        x6 = relu(self.up1(x5, torch.zeros_like(s1)))
        x7 = relu(self.up2(x6, None))
        x_out = self.conv2(x7)
        return x_out




from math import sqrt
class UNet3D3_Res_regul(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros',
                 upsamp_mode='trilinear', dropout=0.0, last_skip=True, decoder_regul=True):
        super(UNet3D3_Res_regul, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode
        self.upsamp_mode = upsamp_mode
        self.max_pool = nn.MaxPool3d((2, 2, 2))
        self.last_skip = last_skip
        self.decoder_regul = decoder_regul

        # use maxpool instead of strided convolution
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*8, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode) #try changing to mid_dim*8

        self.mid_blocks = nn.ModuleList([nn.Conv3d(mid_dim * 8, mid_dim * 8, kernel_size=3, padding=1, stride=1) for _ in range(num_res)])

        # self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*8, swish=swish, padding_mode=padding_mode)
        self.up0 = UpConv3DNew(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up1 = UpConv3DNew(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up2 = UpConv3DNew(mid_dim*2, mid_dim, kernel_size=3, skip=self.last_skip, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.down3 = nn.utils.weight_norm(self.down3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    @staticmethod
    def numel_except(tensor, dim):
        numel = 1
        shape = tensor.shape
        for i in range(len(shape)):
            if i != dim:
                numel *= shape[i]
        return numel

    def forward(self, code_im, regul=False):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = self.conv1(code_im)
        x2 = self.down1(self.max_pool(relu(x1))) #mid*2
        x3 = self.down2(self.max_pool(relu(x2))) #mid*4
        x4 = self.down3(self.max_pool(relu(x3)))

        x_mid = x4
        for res in self.mid_blocks:
            x_mid = res(relu(x_mid))

        if regul:
            norm_sum = torch.norm(x2, dim=1).sum()*sqrt(2)+torch.norm(x3, dim=1).sum()*sqrt(4)\
                       +torch.norm(x4, dim=1).sum()*sqrt(8)+torch.norm(x_mid, dim=1).sum()*sqrt(8)
            numel_sum = self.numel_except(x2.detach(),1) + self.numel_except(x3.detach(),1)\
                        + self.numel_except(x4.detach(),1) + self.numel_except(x_mid.detach(),1)

        x5 = relu(self.up0(x_mid, x3))
        x6 = relu(self.up1(x5, x2))
        x7 = relu(self.up2(x6, x1))

        if hasattr(self, 'decoder_regul') and self.decoder_regul and regul:
            norm_sum += torch.norm(x5, dim=1).sum()*sqrt(8)+torch.norm(x6, dim=1).sum()*sqrt(4)\
                       +torch.norm(x7, dim=1).sum()*sqrt(2)
            numel_sum += self.numel_except(x5.detach(),1) + self.numel_except(x6.detach(),1)\
                        + self.numel_except(x7.detach(),1)

        x = self.conv4(x7)
        if regul: return x, norm_sum / numel_sum
        else: return x



from math import sqrt
class UNet3D3_Res_regul2(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros',
                 upsamp_mode='trilinear', last_skip=True, decoder_regul=True):
        super(UNet3D3_Res_regul2, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode
        self.upsamp_mode = upsamp_mode
        self.max_pool = nn.MaxPool3d((2, 2, 2))
        self.last_skip = last_skip
        self.decoder_regul = decoder_regul

        # use maxpool instead of strided convolution
        self.conv1 = DoubleConv(in_dim, mid_dim*2, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)
        self.down1 = DoubleConv(mid_dim*2, mid_dim*4, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)
        self.down2 = DoubleConv(mid_dim*4, mid_dim*8, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)

        self.skip1 = DoubleConv(mid_dim*2, mid_dim*2, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)
        self.skip2 = DoubleConv(mid_dim*4, mid_dim*4, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)

        self.mid_blocks = nn.ModuleList([DoubleConv(mid_dim*8, mid_dim*8, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm) for _ in range(num_res)])

        # self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*8, swish=swish, padding_mode=padding_mode)
        self.up0 = UpConv3DNew(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up1 = UpConv3DNew(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up2 = UpConv3DNew(mid_dim*2, mid_dim, kernel_size=3, skip=False, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im, regul=False):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im))
        x2 = self.max_pool(relu(self.down1(x1)))
        x3 = self.max_pool(relu(self.down2(x2)))

        s1 = self.skip1(x1)
        s2 = self.skip2(x2)

        x_mid = x3
        for res in self.mid_blocks:
            x_mid = res(x_mid)

        x5 = relu(self.up0(x_mid, s2))
        x6 = relu(self.up1(x5, s1))
        x7 = relu(self.up2(x6, None))

        x = self.conv4(x7)
        if regul: return x, torch.norm(s1, dim=1).mean()+torch.norm(s2, dim=1).mean()+torch.norm(x_mid, dim=1).mean()
        else: return x

if __name__ == "__main__":
    unet = UNet3D3_Res_regul2(32, 64, 64, swish=15).cuda()
    t = torch.rand(2,32,16,32,32).cuda()
    o, norm_reg = unet(t, regul=True)
    # o, norm_regul = unet(t, regul=True)


class UNet3D3_Res_regul_trilin(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros',
                 upsamp_mode='trilinear', decoder_regul=True, noise_mag=None, coords_range=None):
        super(UNet3D3_Res_regul_trilin, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode
        self.upsamp_mode = upsamp_mode
        self.max_pool = nn.MaxPool3d((2, 2, 2))
        self.decoder_regul = decoder_regul
        self.noise_mag = noise_mag
        self.coords_range = coords_range
        self.out_dim = out_dim; self.mid_dim = mid_dim

        # use maxpool instead of strided convolution  # 32 -> 128 256 512
        self.conv1 = DoubleConv(in_dim, mid_dim*2, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)
        self.down1 = DoubleConv(mid_dim*2, mid_dim*4, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)
        self.down2 = DoubleConv(mid_dim*4, mid_dim*8, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)

        self.mid_blocks = nn.ModuleList([DoubleConv(mid_dim*8, mid_dim*8, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm) for _ in range(num_res)])

        # self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*8, swish=swish, padding_mode=padding_mode)
        self.up0 = UpConv3DNew(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up1 = UpConv3DNew(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up2 = UpConv3DNew(mid_dim*2, mid_dim, kernel_size=3, skip=False, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)

        self.conv4 = DoubleConv(mid_dim, out_dim, kernel_size=kernel_size, padding_mode=padding_mode, swish=swish, weight_norm=weight_norm)

    def output_dim(self):
        return self.out_dim + self.mid_dim * (8+4+2)

    @staticmethod
    def numel_except(tensor, dim):
        numel = 1
        shape = tensor.shape
        for i in range(len(shape)):
            if i != dim:
                numel *= shape[i]
        return numel

    def forward(self, code_im, coords=None, regul=False, skip_noise=False):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = self.conv1(code_im)
        x2 = self.down1(self.max_pool(relu(x1))) #mid*2
        x3 = self.down2(self.max_pool(relu(x2))) #mid*4

        x_mid = x3
        for res in self.mid_blocks:
            x_mid = res(relu(x_mid))

        if regul:
            norm_sum = torch.norm(x2, dim=1).mean()+torch.norm(x3, dim=1).mean()+torch.norm(x_mid, dim=1).mean()

        if skip_noise and self.noise_mag is not None:
            s1 = x1 + x1.detach() * torch.randn_like(x1.detach()) * self.noise_mag[0]
            s2 = x2 + x2.detach() * torch.randn_like(x2.detach()) * self.noise_mag[1]
        else: s1 = x1; s2 = x2;

        x5 = relu(self.up0(x_mid, s2))
        x6 = relu(self.up1(x5, s1))
        x7 = relu(self.up2(x6, None))
        x_out = self.conv4(x7)

        if hasattr(self, 'decoder_regul') and self.decoder_regul and regul:
            norm_sum += torch.norm(x5, dim=1).mean()+torch.norm(x6, dim=1).mean()\
                       +torch.norm(x7, dim=1).mean() + torch.norm(x_out,dim=1).mean()

        codes = [x_out,x6,x5,x_mid]
        if coords is not None:
            codes = self.code_sample(coords, codes)
        if regul: return codes, norm_sum
        else: return codes

    def code_sample(self, coords, codes):
        # assuming -1 to 1 range
        # coords: B x N x 1 x 1 x 3
        coords = coords.unsqueeze(2).unsqueeze(2)
        samples = [F.grid_sample(code, coords, padding_mode='border', align_corners=True) for code in codes]
        return torch.cat(samples, dim=1).squeeze(-1).squeeze(-1).permute([0,2,1])  # B x C x N --> B x N x C


def coord_normalize(coords_range, coords):
    # coords: # B x N x 3
    range_max = coords_range
    coords_norm = torch.zeros_like(coords)
    for i in range(len(range_max)):
        coords_norm[:,:,i] = coords[:,:,i] * 2. / range_max[i] - 1.
    assert(coords_norm.max()<1.0);
    assert(coords_norm.min()>-1.0)
    return coords_norm



class UNet3D3_Res_regul_vae(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros',
                 upsamp_mode='trilinear', last_skip=True, decoder_regul=True, determine=False):
        super(UNet3D3_Res_regul_vae, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode
        self.upsamp_mode = upsamp_mode
        self.max_pool = nn.MaxPool3d((2, 2, 2))
        self.last_skip = last_skip
        self.decoder_regul = decoder_regul
        self.determine = determine

        # use maxpool instead of strided convolution
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        self.down1 = ConvVAE(mid_dim, mid_dim*2, kernel_size=3, stride=1, padding_mode=padding_mode, swish=swish)
        self.down2 = ConvVAE(mid_dim*2, mid_dim*4, kernel_size=3, stride=1, padding_mode=padding_mode, swish=swish)
        self.down3 = ConvVAE(mid_dim*4, mid_dim*8, kernel_size=3, stride=1, padding_mode=padding_mode, swish=swish) #try changing to mid_dim*8

        # self.mid_blocks = nn.ModuleList([nn.Conv3d(mid_dim * 8, mid_dim * 8, kernel_size=3, padding=1, stride=1) for _ in range(num_res)])
        self.mid_blocks = nn.ModuleList([DoubleConv(mid_dim * 8, mid_dim * 8, kernel_size=3, padding_mode=padding_mode, swish=swish) for _ in range(num_res)])

        # self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*8, swish=swish, padding_mode=padding_mode)
        self.up0 = UpConv3DNew(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up1 = UpConv3DNew(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)
        self.up2 = UpConv3DNew(mid_dim*2, mid_dim, kernel_size=3, skip=self.last_skip, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode, upsamp_mode=upsamp_mode)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv4)
        self.conv1.remove_weight_norm(); self.down1.remove_weight_norm(); self.down2.remove_weight_norm(); self.down3.remove_weight_norm()
        for res in self.mid_blocks: res.remove_weight_norm()
        self.up0.remove_weight_norm(); self.up1.remove_weight_norm(); self.up2.remove_weight_norm()

    @staticmethod
    def numel_except(tensor, dim):
        numel = 1
        shape = tensor.shape
        for i in range(len(shape)):
            if i != dim:
                numel *= shape[i]
        return numel

    def forward(self, code_im, regul=False):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = self.conv1(code_im)
        # print(x1.shape)
        x2, s2, lv2 = self.down1(self.max_pool(relu(x1))) #mid*2
        # print(x2.shape)
        x3, s3, lv3 = self.down2(self.max_pool(relu(x2))) #mid*4
        x4, s4, lv4 = self.down3(self.max_pool(relu(x3)))

        x_mid = s4
        for res in self.mid_blocks:
            x_mid = res(relu(x_mid))

        if regul:
            norm_sum = torch.norm(x2, dim=1).sum()*sqrt(2)+torch.norm(x3, dim=1).sum()*sqrt(4)\
                       +torch.norm(x4, dim=1).sum()*sqrt(8)+torch.norm(x_mid, dim=1).sum()*sqrt(8)
            numel_sum = self.numel_except(x2.detach(),1) + self.numel_except(x3.detach(),1)\
                        + self.numel_except(x4.detach(),1) + self.numel_except(x_mid.detach(),1)
            vae_regul = (lv2+1).pow(2).mean() + (lv3+1).pow(2).mean() + (lv4+1).pow(2).mean()

        if self.determine:
            x5 = relu(self.up0(x_mid, x3)); x6 = relu(self.up1(x5, x2)); x7 = relu(self.up2(x6, None))
        else:
            x5 = relu(self.up0(x_mid, s3)); x6 = relu(self.up1(x5, s2)); x7 = relu(self.up2(x6, None))

        # if hasattr(self, 'decoder_regul') and self.decoder_regul and regul:
        #     norm_sum += torch.norm(x5, dim=1).sum()*sqrt(8)+torch.norm(x6, dim=1).sum()*sqrt(4)\
        #                +torch.norm(x7, dim=1).sum()*sqrt(2)
        #     numel_sum += self.numel_except(x5.detach(),1) + self.numel_except(x6.detach(),1)\
        #                 + self.numel_except(x7.detach(),1)

        x = self.conv4(x7)
        if regul: return x, norm_sum / numel_sum, vae_regul
        else: return x




class UNet3D3_Mid(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros'):
        super(UNet3D3_Mid, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res
        self.padding_mode = padding_mode

        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*8, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)

        self.mid_blocks = nn.ModuleList([nn.Conv3d(mid_dim * 8, mid_dim * 8, kernel_size=3, padding=1, stride=1) for _ in range(num_res)])

        self.up0 = UpConv3DSimple(mid_dim*8, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*4, swish=swish, padding_mode=padding_mode)
        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)
        self.up2 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=False, weight_norm=weight_norm, swish=swish, padding_mode=padding_mode)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, padding_mode=padding_mode)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.down3 = nn.utils.weight_norm(self.down3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im)) #mid
        x2 = relu(self.down1(x1)) #mid*2
        x3 = relu(self.down2(x2)) #mid*4
        x4 = relu(self.down3(x3))

        for res in self.mid_blocks:
            x4 = res(x4)

        x5 = relu(self.up0(x4, x3))
        x6 = relu(self.up1(x5, x2))
        x7 = relu(self.up2(x6))

        x = self.conv4(x7)

        return x



class UNet3D3_mid_ori(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2, padding_mode='zeros'):
        super(UNet3D3_mid_ori, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res

        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*4, kernel_size=3, padding=1, stride=2)

        self.mid_blocks = nn.ModuleList([nn.Conv3d(mid_dim * 4, mid_dim * 4, kernel_size=3, padding=1, stride=1) for _ in range(num_res)])

        self.up0 = UpConv3DSimple(mid_dim*4, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*4, swish=swish)
        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish)
        self.up2 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.down3 = nn.utils.weight_norm(self.down3)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im)) #mid
        x2 = relu(self.down1(x1)) #mid*2
        x3 = relu(self.down2(x2)) #mid*4
        x4 = relu(self.down3(x3))

        for mid in self.mid_blocks:
            x4 = relu(mid(x4))

        x5 = relu(self.up0(x4, x3))
        x6 = relu(self.up1(x5, x2))
        x7 = relu(self.up2(x6, x1))

        x = self.conv4(x7)

        return x


class JohnsonNet(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5, num_res=2):
        super(JohnsonNet, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.num_res = num_res

        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)

        self.res_blocks = nn.ModuleList([ResidualBlock3D(mid_dim*4,  weight_norm=weight_norm, swish=swish) for i in range(num_res)])

        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=False, weight_norm=weight_norm, swish=swish)
        self.up2 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=False, weight_norm=weight_norm, swish=swish)
        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im)) #mid
        x2 = relu(self.down1(x1)) #mid*2
        x3 = relu(self.down2(x2)) #mid*4

        for res in self.res_blocks:
            x3 = res(x3)

        x4 = relu(self.up1(x3))
        x5 = relu(self.up2(x4))
        x = self.conv4(x5)

        return x

class UNet3D4(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim,  weight_norm=True, swish=None, kernel_size=5):
        super(UNet3D4, self).__init__()
        self.weight_norm = weight_norm
        self.relu = nn.LeakyReLU(0.2)
        self.swish = swish
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*6, kernel_size=3, padding=1, stride=2)
        self.down4 = nn.Conv3d(mid_dim*6, mid_dim*6, kernel_size=3, padding=1, stride=2)

        self.up0 = UpConv3DSimple(mid_dim*6, mid_dim*6, kernel_size=3, skip=True, weight_norm=weight_norm, skip_channels=mid_dim*6, swish=swish)
        self.up1 = UpConv3DSimple(mid_dim*6, mid_dim*4, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish, skip_channels=mid_dim*4)
        self.up2 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish)
        self.up3 = UpConv3DSimple(mid_dim*2, max(out_dim,mid_dim), kernel_size=3, skip=True, weight_norm=weight_norm, swish=swish)

        self.conv4 = nn.Conv3d(max(out_dim,mid_dim), out_dim, kernel_size=kernel_size, padding=kernel_size//2)

        if weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.down1 = nn.utils.weight_norm(self.down1)
            self.down2 = nn.utils.weight_norm(self.down2)
            self.down3 = nn.utils.weight_norm(self.down3)
            self.down4 = nn.utils.weight_norm(self.down4)
            self.conv4 = nn.utils.weight_norm(self.conv4)

    def forward(self, code_im):
        # B x C x H x W
        if hasattr(self, 'swish') and self.swish is not None:
            relu = lambda t: torch.sigmoid(self.swish*t)*t
        else: relu = self.relu

        x1 = relu(self.conv1(code_im)) #mid
        d1 = relu(self.down1(x1)) #mid*2
        d2 = relu(self.down2(d1)) #mid*4
        d3 = relu(self.down3(d2))
        d4 = relu(self.down4(d3))

        u1 = relu(self.up0(d4, d3))
        u2 = relu(self.up1(u1, d2))
        u3 = relu(self.up2(u2, d1))
        u4 = relu(self.up3(u3, x1))

        x = self.conv4(u4)

        return x


class UNet3DInstNorm(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, code_std=1.):
        super(UNet3DInstNorm, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=5, padding=2)
        self.in1 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)
        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.in2 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.in3 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.in4 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)

        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True)
        self.in4 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.up2 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=True)
        self.in5 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, code_im):
        # B x C x H x W
        x1 = self.relu(self.in1(self.conv1(code_im))) #mid
        x2 = self.relu(self.in2(self.down1(x1))) #mid*2
        x3 = self.relu(self.in3(self.down2(x2))) #mid*4

        x4 = self.relu(self.in4(self.up1(x3, x2))) #mid*2
        x5 = self.relu(self.in5(self.up2(x4, x1)))
        x = self.conv4(x5)
        return x

    def kaiming_init(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.down1.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.down2.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.up1.conv1.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.up1.conv2.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.up2.conv1.weight, nonlinearity='leaky_relu', a=0.2)
        nn.init.kaiming_normal_(self.up2.conv2.weight, nonlinearity='leaky_relu', a=0.2)
        # self.conv1 = nn.utils.weight_norm(self.conv1)


class UNet3DInstNorm2(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(UNet3DInstNorm2, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=5, padding=2)
        self.in1 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)

        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.in2 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.in3 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)
        self.in4 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)

        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*4, kernel_size=3, skip=True, skip_channels=mid_dim*4)
        self.up2 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True)
        self.up3 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3, skip=True)
        self.in5 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)
        self.in6 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.in7 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, code_im):
        # B x C x H x W
        x1 = self.relu(self.in1(self.conv1(code_im))) #mid

        x2 = self.relu(self.in2(self.down1(x1))) #mid*2
        x3 = self.relu(self.in3(self.down2(x2))) #mid*4
        x4 = self.relu(self.in4(self.down3(x3))) #mid*4

        x5 = self.relu(self.in5(self.up1(x4, x3))) #mid*2
        x6 = self.relu(self.in6(self.up2(x5, x2)))
        x7 = self.relu(self.in7(self.up3(x6, x1)))
        x = self.conv4(x7)
        return x


class UNet3DInstNorm3(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(UNet3DInstNorm3, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=5, padding=2)
        self.in1 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)

        self.down1 = nn.Conv3d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.down2 = nn.Conv3d(mid_dim*2, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.down3 = nn.Conv3d(mid_dim*4, mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.in2 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.in3 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)
        self.in4 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)

        self.up1 = UpConv3DSimple(mid_dim*4, mid_dim*4, kernel_size=3, skip=True, skip_channels=mid_dim*4)
        self.up2 = UpConv3DSimple(mid_dim*4, mid_dim*2, kernel_size=3, skip=True)
        self.up3 = UpConv3DSimple(mid_dim*2, mid_dim, kernel_size=3)
        self.in5 = torch.nn.InstanceNorm3d(mid_dim*4, affine=True, track_running_stats=True)
        self.in6 = torch.nn.InstanceNorm3d(mid_dim*2, affine=True, track_running_stats=True)
        self.in7 = torch.nn.InstanceNorm3d(mid_dim, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, code_im):
        # B x C x H x W
        x1 = self.relu(self.in1(self.conv1(code_im))) #mid

        x2 = self.relu(self.in2(self.down1(x1))) #mid*2
        x3 = self.relu(self.in3(self.down2(x2))) #mid*4
        x4 = self.relu(self.in4(self.down3(x3))) #mid*4

        x5 = self.relu(self.in5(self.up1(x4, x3))) #mid*2
        x6 = self.relu(self.in6(self.up2(x5, x2)))
        x7 = self.relu(self.in7(self.up3(x6, x1)))
        x = self.conv4(x7)
        return x

import math
class FourierFeature3D(nn.Module):
    def __init__(self, num_freq, cutoff):
        super().__init__()

        freqs = torch.linspace(0, cutoff, num_freq)
        self.register_buffer("freqs", freqs)
        self.num_f = num_freq
        self.out_dim = num_freq * 6 + 3

    def forward(self, coords):
        # coords: B x N x 3
        s = coords.shape
        f = self.freqs  # F
        # print((coords.unsqueeze(-1) * f[None,None,None,:]).shape)
        augmented = (coords.unsqueeze(-1) * f[None,None,None,:]).view(s[0],s[1],-1) * 2 * math.pi
        # B x N x 3 x F --> B x N x (3xF)
        out_coords = torch.cat([coords, torch.sin(augmented), torch.cos(augmented)], -1)
        # B x N x (6xF+3)
        return out_coords

class Decoder_PE(nn.Module):
    def __init__(
        self, num_freq, cutoff,
        latent_size,
        dims,
        input_dim,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        layer_norm=False,
        use_tanh=True,
        soft_plus=-1,
        sine_act=False,
        swish=False,
        swish_beta=1.,
        renormalize=False,
        out_dim=1
    ):
        super().__init__()
        self.pos_enc = FourierFeature3D(num_freq, cutoff)
        self.dec = DecoderSimple(latent_size,dims,self.pos_enc.out_dim,dropout,dropout_prob,norm_layers,latent_in,weight_norm,
                            layer_norm,use_tanh,soft_plus,sine_act,swish,swish_beta,renormalize,out_dim)

    def forward(self, inputs):
        # inputs: B x N x (3+C)
        xyz = inputs[:,:,:3]
        codes = inputs[:,:,3:]
        augmented = self.pos_enc(xyz)
        input_all = torch.cat([augmented, codes], -1)
        return self.dec(input_all)

class DecoderSimple(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        input_dim,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        layer_norm=False,
        use_tanh=True,
        soft_plus=-1,
        sine_act=False,
        swish=False,
        swish_beta=1.,
        renormalize=False,
        out_dim=1
    ):
        super(DecoderSimple, self).__init__()

        dims = [latent_size + input_dim] + dims + [out_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in

        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        # assert(not weight_norm or not layer_norm)

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
            if weight_norm:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))
            if layer_norm and layer in self.norm_layers:
                setattr(self, "layernorm" + str(layer), nn.LayerNorm(out_dim))


        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        if int(soft_plus) == -1: self.soft_plus = None
        else: self.soft_plus = nn.Softplus(beta=soft_plus)

        self.swish = swish
        self.swish_beta = swish_beta
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(0.15)
        self.sine = sine_act

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.renormalize = renormalize

    def weight_2norm(self):
        norm_sum = 0.

    # input: B x N x C
    def forward(self, input):
        x = input
        x_dim = len(x.shape)

        for layer in range(0, self.num_layers - 1):
            relu = self.relu
            if hasattr(self, 'swish') and self.swish:
                if hasattr(self, 'swish_beta'): beta = self.swish_beta
                else: beta = 1
                relu = lambda t: torch.sigmoid(beta*t)*t
            if hasattr(self, 'sine') and self.sine: relu = torch.sin
            if hasattr(self, 'soft_plus') and self.soft_plus and (layer==0 or layer in self.latent_in): relu = self.soft_plus
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], x_dim-1)
            x = lin(x)
            # last layer Tanh
            if layer < self.num_layers - 2:
                x = relu(x)
                if hasattr(self, 'layer_norm') and self.layer_norm and layer in self.norm_layers:
                    layernorm = getattr(self, 'layernorm'+str(layer))
                    x = layernorm(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_tanh:
            x = self.tanh(x)

        return x


# if __name__ == "__main__":
#     unet = UNet3DInstNorm(64,64,64)
#     t = torch.rand(2,64,8,16,16)
#     o=unet(t)
#     zopt = ZOptimizer(2,[3,4,4], 0.01, 1.)
#
#     layers = [256, 256, 256, 256, 256, 256]
#     norm_layers = [0, 1, 3, 4, 5]
#     decoder = DecoderSimple(64, layers, 3, layer_norm=True, norm_layers=norm_layers, latent_in=[3],
#                               swish=True, use_tanh=True, swish_beta=10.)


import math
import torch
from torch.nn.init import _calculate_correct_fan
def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> siren.init.siren_uniform_(w, mode='fan_in', c=6)
    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


# a=torch.rand(1,3,4,4,4)
# conv = torch.nn.Conv3d(3,3,3,1, padding=1, padding_mode='replicate')
# conv_pad=conv(a)
# a_paded = F.pad(a, [1]*6, 'replicate')
# conv_paded = F.conv3d(a_paded,conv.weight,conv.bias,padding=0)
# conv_pad-conv_paded
import os
from canvas import Canvas
import numpy as np
import torch
import random
import math
import torch.nn.functional as F
#test

class Shapes:
    def __init__(self):
        self.num_shapes = 0
        self.shapes = []
        self.canvas_resol = 300
        self.canvas = Canvas((self.canvas_resol, self.canvas_resol))
        self.points = ()
        self.centers = ()
        self.normals = ()
        self.density = 250
        self.torch_points = None
        self.torch_normals = None
        self.perturbed_points = None
        self.perturbed_points_sdf = None
        self.masked_pts = None
        self.masked_sdf = None

    def reset(self):
        self.num_shapes = 1
        self.shapes = []
        self.canvas = Canvas((self.canvas_resol, self.canvas_resol))

    def copy_light(self):
        copy = Shapes()
        copy.num_shapes = self.num_shapes; copy.shapes = self.shapes
        copy.canvas=()
        return copy

    def add_shape(self, shape):
        shape_sdf = shape.computeSDFCanvas(self.canvas)
        if self.check_adding(shape_sdf):
            self.shapes.append(shape)
            if self.num_shapes == 0:
                self.canvas.canvas = shape.computeSDFCanvas(self.canvas)
            else:
                self.canvas.canvas = np.minimum(self.canvas.canvas, shape_sdf)
            self.num_shapes += 1
            return True
        else:
            return False

    def compute_canvas(self):
        self.canvas = Canvas((self.canvas_resol, self.canvas_resol))
        if len(self.shapes) > 0:
            self.canvas.canvas = self.shapes[0].computeSDFCanvas(self.canvas)
            for i in range(len(self.shapes)-1):
                shape_sdf = self.shapes[i+1].computeSDFCanvas(self.canvas)
                self.canvas.canvas = np.minimum(self.canvas.canvas, shape_sdf)

    def randomly_translate(self, offset=0.1):
        for shape in self.shapes:
            center = (np.random.uniform(offset, 1-offset), np.random.uniform(offset, 1-offset))
            shape.center = center

    def sample_points(self, density=None, center_gt=False, add_normal=False):
        x = np.zeros(0)
        y = np.zeros(0)
        x_cent = np.zeros(0)
        y_cent = np.zeros(0)
        nx = np.zeros(0)
        ny = np.zeros(0)
        if density is not None:
            self.density = density
        else: density = self.density

        for shape in self.shapes:
            if add_normal:
                x_tmp, y_tmp, nx_tmp, ny_tmp, x_cent_tmp, y_cent_tmp = shape.sample_points(density, True, True)
                nx = np.concatenate((nx, nx_tmp))
                ny = np.concatenate((ny, ny_tmp))
            else:
                x_tmp, y_tmp, x_cent_tmp, y_cent_tmp = shape.sample_points(density, center_gt=True)
            x_cent = np.concatenate((x_cent, x_cent_tmp))
            y_cent = np.concatenate((y_cent, y_cent_tmp))
            x = np.concatenate((x, x_tmp))
            y = np.concatenate((y, y_tmp))

        self.points = (x, y)
        self.centers = (x_cent, y_cent)
        self.normals = (nx, ny)

        if add_normal: return self.points, self.normals, (x_cent, y_cent)
        else: return self.points, (x_cent, y_cent)

    def sample_points_torch(self, center_out=False, add_normal=False):
        if self.torch_points is not None and center_out and self.centers is not None:
            return self.torch_points, self.centers
        elif self.torch_points is not None and center_out is None:
            return self.torch_points

        import torch
        if self.points and self.centers:
            point, center = self.points, self.centers
        elif add_normal:
            point, normals, center = self.sample_points(center_gt=True, add_normal=True)
        else:
            point, center = self.sample_points(center_gt=True)
        point_x = torch.from_numpy(point[0]).float()
        point_y = torch.from_numpy(point[1]).float()
        points = torch.cat((point_x.unsqueeze(0), point_y.unsqueeze(0)), 0)
        self.torch_points = points

        center_x = torch.from_numpy(center[0]).float()
        center_y = torch.from_numpy(center[1]).float()
        centers = torch.cat((center_x.unsqueeze(0), center_y.unsqueeze(0)), 0)
        self.centers = centers
        if add_normal:
            nx = torch.from_numpy(normals[0]).float()
            ny = torch.from_numpy(normals[1]).float()
            n = torch.stack((nx, ny))
            self.torch_normals = n
            return points, n, centers
        else: return points, centers

    @staticmethod
    def perturb_sample_half_half(pts, noise_size, range_=(0,1)):
        # pts: B x N x 2
        # half and half
        shape = list(pts.shape)
        half = shape[1] // 2
        shape[1] = half
        noise = torch.randn(shape).cuda() * noise_size
        near_surf = pts[:, :half] + noise
        mask = ((near_surf >= range_[0]) & (near_surf <= range_[1])).sum(2) == 2
        all = torch.zeros_like(pts).cuda()
        for b, mask_b in enumerate(mask.cuda()):
            near = near_surf[b, mask_b]
            rand = torch.empty(pts.shape[1]-near.shape[0], 2).uniform_(0,1).cuda()
            all[b] = torch.cat([near, rand], 0)
        return all

    @staticmethod
    def perturb_sample_half_half_trunc(pts, noise_size, range_=(0,1)):
        # pts: B x N x 2
        # half and half
        shape = list(pts.shape)
        half = shape[1] // 2
        shape[1] = half

        theta = torch.empty([shape[0],shape[1],1]).uniform_(0,2*math.pi).cuda()
        noise_dir = torch.cat([torch.cos(theta), torch.sin(theta)], 2).cuda()
        noise_mag = torch.empty([shape[0],shape[1],1]).uniform_(0,1).cuda() * noise_size
        noise = noise_dir * noise_mag
        # return pts + noise
        near_surf = pts[:, :half] + noise
        mask = ((near_surf >= range_[0]) & (near_surf <= range_[1])).sum(2) == 2
        for b, mask_b in enumerate(mask.cuda()):
            near = near_surf[b, mask_b]
            while near.shape != near_surf[0].shape:
                near = torch.cat([near, near[:half-near.shape[0]]], 0)
            # print(near.shape)
            # print(near_surf[0].shape)
            assert(near.shape == near_surf[0].shape)
            near_surf[b] = near
        uni = torch.empty(shape).uniform_(0,1).cuda()
        return torch.cat([near_surf, uni],1)

    @staticmethod
    def perturb_SDF_sample(pts, sdf, range_min, range_max):
        # pts: B x N x 3
        # sdf: B x N
        device = pts.device
        shape = list(pts.shape)
        shape[1] = shape[1] // 2
        num_pts = shape[1]
        noise_dir = F.normalize(torch.randn(shape), dim=2).cuda()  # B x N//2 x 3
        noise_mag = torch.empty([shape[0],shape[1],1]).uniform_(0,1).to(device) * sdf[:,:num_pts].abs().unsqueeze(-1)  # B x N//2 x 1
        noise = noise_dir * noise_mag  # B x N//2 x 3
        near_surf = pts[:,:num_pts,:] + noise  # B x N//2 x 3
        mask = torch.all(near_surf > range_min.to(device), dim=2) & torch.all(near_surf < range_max.to(device), dim=2)  # B x N//2

        for b, mask_b in enumerate(mask.cuda()):
            near = near_surf[b, mask_b]
            while near.shape != near_surf[0].shape:
                near = torch.cat([near, near[:num_pts-near.shape[0]]], 0)
            assert(near.shape == near_surf[0].shape)
            near_surf[b] = near

        uniform = torch.empty(shape).uniform_(0,1) * range_max  # B x N//2 x 3
        out = torch.cat([near_surf, uniform.to(device)],1)
        assert(pts.shape[1] == out.shape[1] or pts.shape[1]-1 == out.shape[1])
        return out

    @staticmethod
    def perturb_sample_trunc3D(pts, noise_size, range_=(1,1,1)):
        # pts: B x N x 3
        shape = list(pts.shape)
        shape[1] = shape[1] // 2
        num_pts = shape[1]
        noise_dir = F.normalize(torch.randn(shape), dim=2).cuda()
        noise_mag = torch.empty([shape[0], shape[1], 1]).uniform_(0, 1).cuda() * noise_size
        noise = noise_dir * noise_mag
        near_surf = pts[:,:num_pts,:] + noise
        mask = ((near_surf[:,:,0]<=range_[0]).float() + (near_surf[:,:,1]<=range_[1]).float()
                + (near_surf[:,:,2]<=range_[2]).float() + (near_surf>=0).sum(dim=2)) == 6
        for b, mask_b in enumerate(mask.cuda()):
            near = near_surf[b, mask_b]
            while near.shape != near_surf[0].shape:
                near = torch.cat([near, near[:num_pts-near.shape[0]]], 0)
            assert(near.shape == near_surf[0].shape)
            near_surf[b] = near
        shape[2] = 1
        uni = torch.cat([torch.empty(shape).uniform_(0,range_[0]), torch.empty(shape).uniform_(0,range_[1]),
                         torch.empty(shape).uniform_(0,range_[2])], 2).cuda()
        out = torch.cat([near_surf, uni],1)
        assert(pts.shape[1] == out.shape[1] or pts.shape[1]-1 == out.shape[1])
        return torch.cat([near_surf, uni],1)


    @staticmethod
    def perturb_sample_trunc3D_equal(pts, noise_size, range_=(1,1,1)):
        # pts: B x N x 3
        shape = list(pts.shape)
        # shape[1] = shape[1] // 2
        num_pts = shape[1]
        noise_dir = F.normalize(torch.randn(shape), dim=2).cuda()
        noise_mag = torch.empty([shape[0], shape[1], 1]).uniform_(0, 1).cuda() * noise_size
        noise = noise_dir * noise_mag
        near_surf = pts[:,:num_pts,:] + noise  # B x N x 3
        for i in range(3):
            near_surf[:,:,i] = torch.clamp(near_surf[:,:,i], min=0., max=range_[i])

        shape[2] = 1
        uni = torch.cat([torch.empty(shape).uniform_(0,range_[0]), torch.empty(shape).uniform_(0,range_[1]),
                         torch.empty(shape).uniform_(0,range_[2])], 2).cuda()  # B x N x 3
        out = torch.cat([near_surf, uni],1)  # B x 2N x 3
        assert(pts.shape[1]*2 == out.shape[1])
        return out


    def perturb_sampled_points(self, num_per_surf_point, sigma, uniform=True, resample=False, range_=(0,1)):
        if self.perturbed_points is not None and not resample:
            return self.perturbed_points
        self.sample_points_torch(add_normal=True)
        surf_points = self.torch_points
        surf_pts_repeat = torch.repeat_interleave(
            surf_points, repeats=num_per_surf_point, dim=1)
        noise = torch.empty(surf_pts_repeat.shape).normal_(mean=0.0, std=sigma)
        self.perturbed_points = surf_pts_repeat + noise
        if uniform:
            self.perturbed_points = torch.cat([
                self.perturbed_points,
                torch.empty((2, surf_points.shape[1]*2)).uniform_(0,1)
            ], 1)
        # TODO: change to masking
        if range_ is not None:
            mask = ((self.perturbed_points >= range_[0]) & (self.perturbed_points <= range_[1])).sum(0) == 2
            # self.perturbed_points = torch.clamp(self.perturbed_points, min=range_[0], max=range_[1])
            self.perturbed_points = self.perturbed_points[:,mask]
            # mask = self.pert
        self.computeSDF_points(self.perturbed_points)
        return self.perturbed_points

    @staticmethod
    def perturb_points(points, num_per_point, sigma, uniform=True, range_=(0,1)):
        surf_pts_repeat = torch.repeat_interleave(
            points, repeats=num_per_point, dim=1)
        noise = torch.empty(surf_pts_repeat.shape).normal_(mean=0.0, std=sigma).cuda()
        perturbed_points = surf_pts_repeat + noise
        if uniform:
            perturbed_points = torch.cat([
                perturbed_points,
                torch.empty_like(points).uniform_(0,1).cuda()
            ], 1)
        if range_ is not None:
            perturbed_points = torch.clamp(perturbed_points, min=range_[0], max=range_[1])
        return perturbed_points


    def mask_surface_sq(self, edge_len=0.0625, mask_surf_pts=False):
        half = edge_len / 2.
        surf_pts = self.torch_points
        center = surf_pts[:, torch.randint(0, surf_pts.shape[1], [1, ])]
        # mask perturbed_pts
        p_pts = self.perturbed_points
        p_sdf = self.perturbed_points_sdf
        mask = ((p_pts >= center - half) & (p_pts <= center + half)).sum(0) == 2
        self.masked_pts = p_pts[:, ~mask]
        self.masked_sdf = p_sdf[~mask]
        # mask surface points
        if mask_surf_pts:
            surf_mask = ((surf_pts >= center - half) & (surf_pts <= center + half)).sum(0) == 2
            self.torch_points = self.torch_points[:, ~surf_mask]
            self.torch_normals = self.torch_normals[:,~surf_mask]
        sdf_canv = self.canvas.canvas; res = self.canvas_resol; vox_len = 1/res
        left_ind = torch.div(center-half, vox_len).clamp(max=res-1, min=0).long()
        right_ind = torch.div(center+half, vox_len).clamp(max=res-1, min=0).long()
        sdf_canv[left_ind[1]:right_ind[1], left_ind[0]:right_ind[0]] = -2.
        return center, edge_len

    def mask_table_intersection(self):
        table_rec = self.shapes[0]
        intersection = []
        for i in range(1, len(self.shapes)):
            shape = self.shapes[i]
            if shape.name == 'rectangle':
                intersection.append([shape.center[0]-shape.length[0]/2, shape.center[0]+shape.length[0]/2])

        margin = 0.005

        surf_y = table_rec.center[1] - table_rec.length[1] / 2

        for inter in intersection:
            surf_pts = self.torch_points
            surf_mask = ((surf_pts >= torch.tensor([inter[0], surf_y-margin]).unsqueeze(1)) &
                         (surf_pts <= torch.tensor([inter[1], surf_y+margin]).unsqueeze(1))).sum(0) == 2
            self.torch_points = self.torch_points[:, ~surf_mask]
            self.torch_normals = self.torch_normals[:, ~surf_mask]

    def has_rectangle(self):
        for i in range(1, len(self.shapes)):
            shape = self.shapes[i]
            if shape.name == 'rectangle':
                return True
        return False


    @staticmethod
    def mask_out_pts_sq(points, center, edge_len, mask_dim, mask_out=False):
        half = edge_len / 2.
        mask = ((points >= center.cuda() - half) & (points <= center.cuda() + half)).sum(2) == 2
        mask = mask.squeeze(0)
        if mask_out:
            return ~mask
        else:
            return points[:, ~mask]
        # if mask_dim == 1: return points[:, ~mask]

    @staticmethod
    def mask_out_near_floor(points, floor_z, norm_thresh, dist_thresh):
        # mask out regions with normal facing downward (-z direction) and near the floor
        # points: N x 6
        pts = points[:,:3]
        normals = points[:,3:]
        mask = (normals[:,2] < norm_thresh) & (pts[:,2] <= floor_z + dist_thresh)
        remainder_points = points[~mask]
        masked_points = points[mask]

        return remainder_points, masked_points

    @staticmethod
    def mask_out_floor_points_z(floor_points, masked_pts, square_size):
        def remove_within_range_xy(pts, center, half):
            pts_xy = pts[:,:2]
            mask_xy = ((pts_xy<(center+half)) & (pts_xy>(center-half))).sum(1)==2
            return pts[~mask_xy]

        while len(masked_pts)>0:
            xy = masked_pts[0,:2].unsqueeze(0)
            floor_points = remove_within_range_xy(floor_points, xy, square_size/2.)
            masked_pts = remove_within_range_xy(masked_pts, xy, square_size/2.)
        return floor_points

    @staticmethod
    def get_depthmap_no_occlusion(verts, camera=(0.332507, -0.489823,1.0167), lookat=(0.5,0.5,0.)):
        vec = torch.tensor(camera) - torch.tensor(lookat)
        vec = vec.unsqueeze(0)
        normal = verts[:, 3:]
        dot_prod = (normal * vec).sum(1)
        valid = dot_prod > 0.
        verts_valid = verts[valid]
        return verts_valid

    @staticmethod
    def get_depthmap_with_occlusion(verts, t, R, resol=(512,512), ranges=([-1,1],[-1,1]), threshold=0.01, noise_std=None, out_ind=False):
        # t: 3 tensor
        # R: 3x3
        T = torch.cat([torch.cat([R, t], -1), torch.tensor([[0., 0., 0., 1.]])], 0)  # camera to world
        T = torch.inverse(T)
        v = verts[:, :3]
        n = verts[:, 3:]
        v_homo = torch.cat([v, torch.ones(v.shape[0], 1)], 1).permute([1, 0])  # 4 x N
        n_homo = torch.cat([n, torch.zeros(n.shape[0], 1)], 1).permute([1, 0])  # 4 x N
        v_cam = torch.matmul(T, v_homo)
        n_cam = torch.matmul(T, n_homo)
        depthmask, depthmap, pts_vis = Shapes.splat_depth_XZ(v_cam, n_cam, resol, ranges, threshold, noise_std=noise_std)
        if noise_std is not None:
            pts_selected_world = torch.matmul(torch.inverse(T), pts_vis).permute([1, 0])[:, :3]
            verts_visible = torch.cat([pts_selected_world, n[depthmask]], 1)
        else: verts_visible = verts[depthmask]
        if out_ind: return depthmask
        else: return verts_visible

    @staticmethod
    def splat_depth_XZ(points, normals, resol, ranges, threshold=0.01, noise_std=None):
        # points, normals: 4 x N
        x = points[0]
        y = points[1]
        z = points[2]
        n_y = normals[1]
        x_range = -ranges[0][0] + ranges[0][1]
        z_range = -ranges[1][0] + ranges[1][1]
        x_r = (x - ranges[0][0]) / x_range
        z_r = (z - ranges[1][0]) / z_range
        x_pixel = 1. / resol[0]
        z_pixel = 1. / resol[1]
        x_ind = (x_r / x_pixel).long()
        z_ind = (z_r / z_pixel).long()
        ind_flat = x_ind + resol[0] * z_ind
        depthmap = torch.ones(resol[0] * resol[1]) * 10
        for k, p in enumerate(points.permute([1, 0])):
            yy = p[1]
            ind = ind_flat[k].item()
            depthmap[ind] = min(depthmap[ind].item(), yy)

        depth_at = torch.index_select(depthmap, 0, ind_flat)
        depth_diff = (depth_at - y).abs()
        selected = (depth_diff < threshold) & (n_y < 0.)

        pts_selected = points[:, selected]
        if noise_std is not None:
            noise = torch.ones(pts_selected.shape[1]).normal_(std=noise_std) * pts_selected[1]
            pts_selected[1] += noise

        return selected, depthmap, pts_selected

    @staticmethod
    def mask_out_pts_cube(pts_norms, center, edge_len):
        # pts_norms: N x 6
        half = edge_len / 2.
        points = pts_norms[:,:3]
        mask = ((points >= center - half) & (points <= center + half)).sum(1) == 3
        return pts_norms[~mask]

    @staticmethod
    def insert_cuboid_pts(xyz_range, num_samples, bias_factor=1):
        (x_min, x_max, y_min, y_max, z_min, z_max) = xyz_range
        x_length = x_max - x_min; y_length = y_max - y_min; z_length = z_max - z_min
        area_floor = x_length * y_length
        area_xz = x_length * z_length
        area_yz = y_length * z_length

        num_xy = int(num_samples * area_floor / (area_floor + area_xz + area_yz) / 2.)
        num_xz = int(num_samples * area_xz / (area_floor + area_xz + area_yz) / 2.)
        num_yz = int(num_samples * area_yz / (area_floor + area_xz + area_yz) / 2.)

        x_samples = torch.rand(num_xy*2, 1) * x_length + x_min
        y_samples = torch.rand(num_xy*2, 1) * y_length + y_min
        z_samples = torch.rand(num_xy*2, 1) * z_length + z_min

        #introducing bias to the sampling of top and bottom surfaces of the floor
        # z_const = torch.cat([torch.ones(num_xy,1)*z_max, torch.ones(num_xy,1)*z_min],0)
        z_const = torch.cat([torch.ones(num_xy*2-num_xy//bias_factor,1)*z_max, torch.ones(num_xy//bias_factor,1)*z_min],0)
        y_const = torch.cat([torch.ones(num_xz,1)*y_max, torch.ones(num_xz,1)*y_min],0)
        x_const = torch.cat([torch.ones(num_yz,1)*x_max, torch.ones(num_yz,1)*x_min],0)

        xy_samples = torch.cat([x_samples, y_samples, z_const], 1)
        xz_samples = torch.cat([x_samples[:2*num_xz], y_const, z_samples[:2*num_xz]], 1)
        yz_samples = torch.cat([x_const, y_samples[:2*num_yz], z_samples[:2*num_yz]], 1)
        pts_samples = torch.cat([xy_samples, xz_samples, yz_samples], 0)

        xy_normal = torch.cat([torch.zeros(num_xy*2,2), torch.ones(num_xy*2,1)],1)
        xy_normal = torch.cat([xy_normal[:num_xy*2-num_xy//bias_factor], -xy_normal[:num_xy//bias_factor]], 0)
        # xy_normal = torch.cat([torch.zeros(num_xy,2), torch.ones(num_xy,1)],1)
        # xy_normal = torch.cat([xy_normal, -xy_normal], 0)
        xz_normal = torch.cat([torch.zeros(num_xz,1), torch.ones(num_xz,1), torch.zeros(num_xz,1)],1)
        xz_normal = torch.cat([xz_normal, -xz_normal])
        yz_normal = torch.cat([torch.ones(num_yz,1), torch.zeros(num_yz,2)],1); yz_normal=torch.cat([yz_normal, -yz_normal], 0)
        normal_samples = torch.cat([xy_normal, xz_normal, yz_normal], 0)

        return torch.cat([pts_samples, normal_samples], 1)


    def uniform_random_points(self, number):
        pts = torch.empty((2, number)).uniform_(0,1)

    def computeSDF_points(self, points):
        # expect 2 x N
        if type(points) == type(torch.empty(0)):
            points_np = points.detach().cpu().numpy()
        else: points_np = points

        sdf = np.ones((points_np.shape[1])) * 10

        for shape in self.shapes:
            sdf_tmp = shape.computeSDF(points_np)
            sdf = np.minimum(sdf_tmp, sdf)
        self.perturbed_points_sdf = torch.from_numpy(sdf)
        return self.perturbed_points_sdf

    def check_adding(self, shape_sdf):
        current = self.canvas.canvas
        return np.sum((np.sign(current) + np.sign(shape_sdf)) < -1) == 0

    def show_canvas(self):
        self.canvas.show_canvas()

    def clear_canvas(self):
        self.canvas.clear()

    def display_points(self):
        self.sample_points(self.density)
        self.canvas.show_points(self.points[0], self.points[1])

    def display_any_points(self, points):
        self.canvas.show_points(points[0], points[1], color='g')

    def display_torch_points(self, points, color=None, hardmax=None, cmap=False):
        if color is None:
            if hardmax is None:
                self.canvas.show_points(points[0].detach().cpu().numpy(),
                                        points[1].detach().cpu().numpy(), color='g')
            else:
                rand_color = Canvas.assign_color_hardmax(hardmax.detach().cpu())
                self.canvas.show_points_color(points[0].detach().cpu().numpy(),
                                              points[1].detach().cpu().numpy(),
                                              c=rand_color.numpy())
        elif type(color) == type(''):
            self.canvas.show_points(points[0].detach().cpu().numpy(),
                                    points[1].detach().cpu().numpy(), color=color)
        else:
            if cmap:
                self.canvas.show_points_cmap(points[0].detach().cpu().numpy(),
                                              points[1].detach().cpu().numpy(),
                                              c=color.detach().cpu().numpy())
            else:
                self.canvas.show_points_color(points[0].detach().cpu().numpy(),
                                             points[1].detach().cpu().numpy(),
                                             c=color.detach().cpu().numpy())

    def draw_point_center_lines(self):
        pass


class Shape:
    def __init__(self, name, center):
        self.name = name
        self.center = center

    def computeSDF(self, points):
        # points being 2 X N numpy array
        raise NotImplementedError("Please Implement this method")
        pass

    def computeSDFCanvas(self, canvas):
        raise NotImplementedError("Please Implement this method")
        pass

    def get_name(self):
        return self.name

    def sample_points(self, density, center_gt=False, add_normal=False):
        # number of points per unit length
        # also return ground truth center of the shape a point belongs to
        raise NotImplementedError("Please Implement this method")
        pass

    def get_length(self):
        raise NotImplementedError("Please Implement this method")
        pass

    @staticmethod
    def compute_grid(canvas_shape, range=((0.,1.), (0.,1.))):
        x_dim = canvas_shape[1]
        y_dim = canvas_shape[0]
        x_lin = np.linspace(range[0][0],range[0][1],x_dim,endpoint=False) + 1/x_dim*0.5
        y_lin = np.linspace(range[1][0],range[1][1],y_dim,endpoint=False) + 1/y_dim*0.5
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        return x_grid, y_grid

    @staticmethod
    def visualize_2D_SDF(decoder, z, res=300, save_fname=None):
        # z is single vector: [1,latent_size]
        if len(z.shape) == 1:
            z_test = z.unsqueeze(0).detach().clone()
        else: z_test = z.clone()
        sdf_im = Shape.compute2D_SDF_single(decoder, z_test[0], res)

        if z_test.shape[0] > 1:
            for zz in z_test[1:]:
                sdf_im = torch.min(sdf_im, Shape.compute2D_SDF_single(decoder, zz, res))

        Canvas.show_image(sdf_im.cpu().numpy())
        if save_fname is not None:
            Canvas.save_image(sdf_im.cpu().numpy(), save_fname)

    @staticmethod
    def visualize_2D_SDF_Trans(decoder, z, translation, res=300, save_fname=None):
        # z is single vector: [1,latent_size] or [B, latent_size]
        # trans is [2] or [B, 2]
        if len(z.shape) == 1:
            z_test = z.unsqueeze(0).detach().clone()
        else: z_test = z.detach().clone()
        if len(translation.shape)==1: translation = translation.unsqueeze(0)
        sdf_im = Shape.compute2D_SDF_single(decoder, z_test[0], res, trans=translation[0])

        if z_test.shape[0] > 1:
            for i, zz in enumerate(z_test[1:]):
                sdf_im = torch.min(sdf_im, Shape.compute2D_SDF_single(
                    decoder, zz, res, trans=translation[i+1]))

        Canvas.show_image(sdf_im.cpu().numpy())
        if save_fname is not None:
            Canvas.save_image(sdf_im.cpu().numpy(), save_fname)


    @staticmethod
    def visualize_2D_SDF_Trans_attentionSDF(decoder, z, res=300, save_fname=None):
        # z: B x K x V



        sdf_im = Shape.compute2D_SDF_single(decoder, z_test[0], res, trans=translation[0])

        if z_test.shape[0] > 1:
            for i, zz in enumerate(z_test[1:]):
                sdf_im = torch.min(sdf_im, Shape.compute2D_SDF_single(
                    decoder, zz, res, trans=translation[i+1]))

        Canvas.show_image(sdf_im.cpu().numpy())
        if save_fname is not None:
            Canvas.save_image(sdf_im.cpu().numpy(), save_fname)


    @staticmethod
    def display_torch_points(points, color=None):
        if color is None:
            Canvas.show_points(points[0].detach().cpu().numpy(),
                                    points[1].detach().cpu().numpy(), color='g')

    @staticmethod
    def compute2D_SDF_single(decoder, z, res=300, trans=None):
        # z is single vector: [latent_size]
        # trans is [x_trans, y_trans]
        x, y = Shape.compute_grid((res, res))
        x_t = torch.from_numpy(x).cuda().float()
        y_t = torch.from_numpy(y).cuda().float()
        x_flat = torch.reshape(x_t, [res * res, 1])
        y_flat = torch.reshape(y_t, [res * res, 1])
        if trans is not None: x_flat += trans[0]; y_flat += trans[1]
        xy = torch.cat((x_flat, y_flat), 1)
        z_copy = z.detach().unsqueeze(0)
        z_repeat = z_copy.repeat(res*res, 1)
        sdf = decoder(torch.cat([xy, z_repeat], 1))
        sdf_im = sdf.detach().reshape([res, res])#.cpu().numpy()
        return sdf_im

    @staticmethod
    def grid_coords(res=300):
        x, y = Shape.compute_grid((res, res))
        x_t = torch.from_numpy(x).cuda().float()
        y_t = torch.from_numpy(y).cuda().float()
        x_flat = torch.reshape(x_t, [res * res, 1])
        y_flat = torch.reshape(y_t, [res * res, 1])
        xy = torch.cat((x_flat, y_flat), 1)
        return xy

    @staticmethod
    def length_xy(x, y):
        return np.sqrt(x**2 + y**2)

    def get_center(self):
        return self.center


def writePly(f_out, vertex):
    write_normal = len(vertex[0]) == 6
    with open(f_out, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(vertex)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if write_normal:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")
        for i in range(len(vertex)):
            v = vertex[i]
            # col =[50 * (x + 1) for x in [ label[i] % 3, label[i] % 4, label[i] % 5 ]]
            if write_normal:
                f.write('{} {} {} {} {} {}\n'
                    .format(v[0], v[1], v[2],
                        v[3], v[4], v[5]))
            else:
                f.write('{} {} {}\n'
                    .format(v[0], v[1], v[2]))


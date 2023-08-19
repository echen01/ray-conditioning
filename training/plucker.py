"""For general notes on Plucker coordinates:
https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf"""

import torch

from torch.nn import functional as F


def plucker_embedding(H, W, intrinsics, c2w, jitter=False):
    """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    c2w: (B, 4, 4)
    intrinsics: (B, 3, 3)
    """    
    cam_pos, ray_dirs = get_rays(H, W, intrinsics, c2w, jitter=jitter)
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)

    plucker = plucker.view(-1, H, W, 6).permute(0, 3, 1, 2)
    return plucker  # (B, 6, H, W, )


def two_plane_embedding(H, W, intrinsics, c2w):
    """Computes the two plane coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    c2w: (B, 4, 4)
    intrinsics: (B, 3, 3)
    """
    B = c2w.shape[0]
    cam_pos, ray_dirs = get_rays(H, W, intrinsics, c2w)
    # cam_pos = cam_pos.view(B * H * W, 3)
    # ray_dirs = ray_dirs.view(B * H * W, 3)
    n = torch.tensor([0, 0, 1.0], device=c2w.device)
    uv = intersect_plane(cam_pos, ray_dirs, n, -1)
    st = intersect_plane(cam_pos, ray_dirs, n, 1)
    uvst = torch.cat([uv, st], dim=-1).view(B, H * W, 6)

    uvst = uvst.view(-1, H, W, 6).permute(0, 3, 1, 2)
    return uvst  # (B, 6, H, W, )


def origin_dir_embedding(H, W, intrinsics, c2w, jitter=False):
    cam_pos, ray_dirs = get_rays(H, W, intrinsics, c2w, jitter=jitter)
    coords = torch.cat([cam_pos, ray_dirs], dim=-1)
    coords = coords.view(-1, H, W, 6).permute(0, 3, 1, 2)
    return coords


def get_rays(H, W, intrinsics, c2w, jitter=False):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=c2w.device),
        torch.arange(H, device=c2w.device),
        indexing="ij",
    )
    B = c2w.shape[0]
    u, v = u.reshape(-1), v.reshape(-1)
    u_noise = v_noise = 0.5
    if jitter:
        u_noise = torch.rand(u.shape, device=c2w.device)
        v_noise = torch.rand(v.shape, device=c2w.device)
    u, v = u + u_noise, v + v_noise  # add half pixel
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
    pixels = pixels.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)
    if intrinsics.sum() == 0:
        inv_intrinsics = torch.eye(3, device=c2w.device).tile(B, 1, 1)
    else:
        inv_intrinsics = torch.linalg.inv(intrinsics)
    rays_d = inv_intrinsics @ pixels  # (B, 3, H*W)
    rays_d = c2w[:, :3, :3] @ rays_d
    rays_d = rays_d.transpose(-1, -2)  # (B, H*W, 3)
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = c2w[:, :3, 3].reshape((-1, 3))  # (B, 3)
    rays_o = rays_o.unsqueeze(1).repeat(1, H * W, 1)  # (B, H*W, 3)

    return rays_o, rays_d


def get_rays_single_image(H, W, intrinsics, c2w):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=c2w.device), torch.arange(H, device=c2w.device)
    )

    u = u.reshape(-1) + 0.5  # add half pixel
    v = v.reshape(-1) + 0.5
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
    if intrinsics.sum() == 0:
        ray_d = pixels
    else:
        rays_d = torch.linalg.inv(intrinsics[:3, :3]) @ pixels
    rays_d = c2w[:3, :3] @ rays_d
    rays_d = rays_d.T  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = torch.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    return rays_o, rays_d


def dot(a, b, axis=-1):
    return torch.sum(a * b, dim=axis)


def intersect_plane(rays_o, rays_d, normal, distance):
    o_dot_n = dot(rays_o, normal)
    d_dot_n = dot(rays_d, normal)
    t = (distance - o_dot_n) / (d_dot_n)
    loc = rays_o + t[..., None] * rays_d
    return loc


def intersect_sphere(ray_o, ray_d):
    """
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    """
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1.0 / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.0).any():
        raise Exception(
            "Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!"
        )
    d2 = torch.sqrt(1.0 - p_norm_sq) * ray_d_cos

    return d1 + d2

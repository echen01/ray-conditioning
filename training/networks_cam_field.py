# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.optimize
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from training.camera_utils import UniformSpherePoseSampler
from training.networks_stylegan2 import (
    SynthesisNetwork,
    normalize_2nd_moment,
    FullyConnectedLayer,
)
from training.encodings import GaussianFourierFeatureTransform

from training.lie import SE3_to_se3


@persistence.persistent_class
class SGC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, lr_multiplier=0.01):
        super().__init__()
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = FullyConnectedLayer(
            in_dim, out_dim, activation="lrelu", lr_multiplier=lr_multiplier
        )

    def forward(self, x, adj, k=2):
        adj = torch.matrix_power(adj, k)
        return self.W(adj @ x)


@persistence.persistent_class
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_dim, out_dim, lr_multiplier=0.01):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = FullyConnectedLayer(
            in_dim, out_dim, activation="lrelu", lr_multiplier=lr_multiplier
        )

    def forward(self, x, adj):
        return self.W(adj @ x)


@persistence.persistent_class
class Processor(MessagePassing):
    """
    Residually processes one iteration of a graph
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        out_dim,
        num_layers=2,
        lr_multiplier=0.01,
    ):
        super().__init__(aggr="mean")  # TODO: Experiment with aggregation schemes
        self.n_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        edge_features = [node_dim * 2 + edge_dim] + [out_dim] * self.num_layers
        node_features = [node_dim * 2] + [out_dim] * self.num_layers

        for idx, in_features, out_features in zip(
            range(num_layers), edge_features[:-1], edge_features[1:]
        ):
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation="lrelu",
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"e_enc{idx}", layer)

        for idx, in_features, out_features in zip(
            range(num_layers), node_features[:-1], node_features[1:]
        ):
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation="lrelu",
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"n_enc{idx}", layer)

    def forward(self, graph):
        edge_index = graph.edge_index
        # cat features together (eij,vi,ei)
        x_receiver = torch.gather(
            graph.x, 0, edge_index[0, :].unsqueeze(-1).repeat(1, graph.x.shape[1])
        )
        x_sender = torch.gather(
            graph.x, 0, edge_index[1, :].unsqueeze(-1).repeat(1, graph.x.shape[1])
        )
        edge_features = torch.cat([x_receiver, x_sender, graph.edge_attr], dim=-1)

        # edge processor
        for idx in range(self.num_layers):
            edge_features = getattr(self, f"e_enc{idx}")(edge_features)

        # aggregate edge_features
        node_features = self.propagate(edge_index, x=graph.x, edge_attr=edge_features)
        # cat features for node processor (vi,\sum_eij)
        features = torch.cat([graph.x, node_features[:, self.out_dim :]], dim=-1)
        # node processor and update graph

        for idx in range(self.num_layers):
            features = getattr(self, f"n_enc{idx}")(features)
        graph.x = features
        graph.edge_attr = edge_features
        return graph

    def message(self, x_i, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=-1)
        return z


# ----------------------------------------------------------------------------
@persistence.persistent_class
class GCNMappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=2,
        num_steps=8,
        lr_multiplier=0.01,
        w_avg_beta=0.998,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.n_vertices = 250
        self.sampler = UniformSpherePoseSampler
        self.cam_radius = 1.3
        self.n_dim = 16
        self.num_steps = num_steps
        # edge_features = [self.n_dim] + [self.w_dim] * self.num_layers
        # node_features = [self.n_dim + self.z_dim] + [self.w_dim] * self.num_layers

        for idx in range(self.num_steps):
            if idx == 0:
                p = GraphConvolution(
                    self.z_dim + self.z_dim,
                    self.w_dim,
                    lr_multiplier=lr_multiplier,
                )
            else:
                p = GraphConvolution(
                    self.w_dim, self.w_dim, lr_multiplier=lr_multiplier
                )
            setattr(self, f"p{idx}", p)

        self.enc = GaussianFourierFeatureTransform(6, 256)
        self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        return_graph=False,
    ):
        in_ids = [self.n_vertices * i for i in range(z.shape[0])]
        z = normalize_2nd_moment(z)
        if c is not None:
            # c = normalize_2nd_moment(c)
            x, adj = self.generate_graph(z, c)
        else:
            x, adj = self.generate_graph(z)

        # Normalize adjacency matrix
        diag = torch.diag(torch.sum(adj, dim=1))
        D = 1 / diag

        if c.sum() != 0:
            adj = D @ adj

        # graph = self.encoder(graph)
        for idx in range(self.num_steps):
            x = getattr(self, f"p{idx}")(x, adj)

        ws = x[in_ids]

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(ws.mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        ws = ws.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            ws[:, :truncation_cutoff] = self.w_avg.lerp(
                ws[:, :truncation_cutoff], truncation_psi
            )
        if return_graph:
            return ws, x, adj
        else:
            return ws

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"

    def generate_graph(self, z, c=None):
        """
        z Tensor[B, self.z_dim]
        c Tensor[B, self.c_dim]

        """
        B = z.shape[0]
        N = self.n_vertices
        z = z.repeat(self.n_vertices, 1)  # [B * N, z_dim]
        in_ids = [i * N for i in range(B)]  # [B]

        poses = self.sampler.sample(
            radius=self.cam_radius,
            batch_size=N * B,
            device=z.device,
        )
        if c is not None:
            poses[in_ids] = c.view(B, 4, 4)
        positions = poses[:, :3, -1].view(B, N, 3).cpu().numpy()

        # TODO Parallelize
        adj_mat = torch.zeros((B * N, B * N), device=z.device)

        for i in range(B):
            tri = ConvexHull(positions[i])

            x1, x2, x3 = tri.simplices.T + N * i
            adj_mat[x1, x2] = 1
            adj_mat[x2, x1] = 1
            adj_mat[x1, x3] = 1
            adj_mat[x3, x1] = 1
            adj_mat[x2, x3] = 1
            adj_mat[x3, x2] = 1
        # poses = poses[:, :3]
        # poses = SE3_to_se3(poses)
        # poses = self.enc(poses)
        # Eposes = normalize_2nd_moment(poses)
        poses = poses.view(-1, 16)
        x = torch.cat([z, poses], dim=-1)
        return x, adj_mat


# ----------------------------------------------------------------------------
@persistence.persistent_class
class LookAtMappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=2,
        num_steps=2,
        lr_multiplier=0.01,
        w_avg_beta=0.998,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.n_vertices = 250
        self.sampler = UniformSpherePoseSampler
        self.cam_radius = 1.3
        self.n_dim = 3
        self.num_steps = num_steps
        # edge_features = [self.n_dim] + [self.w_dim] * self.num_layers
        # node_features = [self.n_dim + self.z_dim] + [self.w_dim] * self.num_layers

        for idx in range(self.num_steps):
            if idx == 0:
                p = Processor(
                    self.z_dim + self.n_dim,
                    self.n_dim + 1,
                    self.w_dim,
                    lr_multiplier=lr_multiplier,
                )
            else:
                p = Processor(
                    self.w_dim, self.w_dim, self.w_dim, lr_multiplier=lr_multiplier
                )
            setattr(self, f"p{idx}", p)

        self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        return_graph=False,
    ):
        in_ids = [self.n_vertices * i for i in range(z.shape[0])]
        z = normalize_2nd_moment(z)
        if c is not None:
            # c = normalize_2nd_moment(c)
            c = c[:, :16]
            graph = self.generate_graph(z, c)
        else:
            graph = self.generate_graph(z)

        # graph = self.encoder(graph)
        for idx in range(self.num_steps):
            graph = getattr(self, f"p{idx}")(graph)

        ws = graph.x[in_ids]

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(ws.mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        ws = ws.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            ws[:, :truncation_cutoff] = self.w_avg.lerp(
                ws[:, :truncation_cutoff], truncation_psi
            )
        if return_graph:
            return ws, graph
        else:
            return ws

    def encoder(self, graph):
        for idx in range(self.num_layers):
            graph.x = getattr(self, f"n_enc{idx}")(graph.x)
            graph.edge_attr = getattr(self, f"e_enc{idx}")(graph.edge_attr)
        return graph

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"

    def generate_graph(self, z, c=None):
        """
        z Tensor[B, self.z_dim]
        c Tensor[B, self.c_dim]

        """
        B = z.shape[0]
        N = self.n_vertices
        z = z.repeat(self.n_vertices, 1)  # [B * N, z_dim]
        in_ids = [i * N for i in range(B)]  # [B]

        poses = self.sampler.sample(
            radius=self.cam_radius,
            batch_size=N * B,
            device=z.device,
        )
        if c is not None:
            poses[in_ids] = c.view(B, 4, 4)
        positions = poses[:, :3, -1].view(B, N, 3).cpu().numpy()
        look_ats = -poses[:, :3, -1].view(B * N, 3)

        # TODO Parallelize
        adj_lists = []
        for i in range(B):
            tri = ConvexHull(positions[i])
            adj_list = (
                self.simplices_to_adj_list(
                    torch.tensor(tri.simplices, dtype=int, device=z.device)
                )
                + N * i
            )
            adj_lists.append(adj_list)
        adj_lists = torch.cat(adj_lists, dim=-1)

        # Compute relative poses
        x1 = look_ats[adj_lists[0]]
        x2 = look_ats[adj_lists[1]]
        rel_pose = x2 - x1
        dist = torch.linalg.norm(rel_pose, dim=-1, keepdim=True)
        edge_feats = torch.cat([rel_pose, dist], dim=-1)
        x = torch.cat([z, look_ats], dim=-1)
        return Data(x=x, edge_index=adj_lists, edge_attr=edge_feats)

    def simplices_to_adj_list(self, simplices):
        adj_mat = torch.zeros(
            (self.n_vertices, self.n_vertices), dtype=int, device=simplices.device
        )
        x1, x2, x3 = simplices.T

        adj_mat[x1, x2] = 1
        adj_mat[x2, x1] = 1
        adj_mat[x1, x3] = 1
        adj_mat[x3, x1] = 1
        adj_mat[x2, x3] = 1
        adj_mat[x3, x2] = 1
        adj_list = adj_mat.nonzero().t().contiguous()
        return adj_list


# ----------------------------------------------------------------------------
@persistence.persistent_class
class GraphMappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=2,
        num_steps=2,
        lr_multiplier=0.01,
        w_avg_beta=0.998,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.n_vertices = 250
        self.sampler = UniformSpherePoseSampler
        self.cam_radius = 1.3
        self.n_dim = 16
        self.num_steps = num_steps
        # edge_features = [self.n_dim] + [self.w_dim] * self.num_layers
        # node_features = [self.n_dim + self.z_dim] + [self.w_dim] * self.num_layers

        for idx in range(self.num_steps):
            if idx == 0:
                p = Processor(
                    self.z_dim + self.n_dim,
                    self.n_dim,
                    self.w_dim,
                    lr_multiplier=lr_multiplier,
                )
            else:
                p = Processor(
                    self.w_dim, self.w_dim, self.w_dim, lr_multiplier=lr_multiplier
                )
            setattr(self, f"p{idx}", p)

        self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        return_graph=False,
    ):
        in_ids = [self.n_vertices * i for i in range(z.shape[0])]
        z = normalize_2nd_moment(z)
        if c is not None:
            # c = normalize_2nd_moment(c)
            c = c[:, :16]
            graph = self.generate_graph(z, c)
        else:
            graph = self.generate_graph(z)

        # graph = self.encoder(graph)
        for idx in range(self.num_steps):
            graph = getattr(self, f"p{idx}")(graph)

        ws = graph.x[in_ids]

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(ws.mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        ws = ws.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            ws[:, :truncation_cutoff] = self.w_avg.lerp(
                ws[:, :truncation_cutoff], truncation_psi
            )
        if return_graph:
            return ws, graph
        else:
            return ws

    def encoder(self, graph):
        for idx in range(self.num_layers):
            graph.x = getattr(self, f"n_enc{idx}")(graph.x)
            graph.edge_attr = getattr(self, f"e_enc{idx}")(graph.edge_attr)
        return graph

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"

    def generate_graph(self, z, c=None):
        """
        z Tensor[B, self.z_dim]
        c Tensor[B, self.c_dim]

        """
        B = z.shape[0]
        N = self.n_vertices
        z = z.repeat(self.n_vertices, 1)  # [B * N, z_dim]
        in_ids = [i * N for i in range(B)]  # [B]

        poses = self.sampler.sample(
            radius=self.cam_radius,
            batch_size=N * B,
            device=z.device,
        )
        if c is not None:
            poses[in_ids] = c.view(B, 4, 4)
        positions = poses[:, :3, -1].view(B, N, 3).cpu().numpy()
        poses = poses.view(B * N, 16)

        # TODO Parallelize
        adj_lists = []
        for i in range(B):
            tri = ConvexHull(positions[i])
            adj_list = (
                self.simplices_to_adj_list(
                    torch.tensor(tri.simplices, dtype=int, device=z.device)
                )
                + N * i
            )
            adj_lists.append(adj_list)
        adj_lists = torch.cat(adj_lists, dim=-1)

        # Compute relative poses
        C1 = poses[adj_lists[0]].view(-1, 4, 4)
        C2 = poses[adj_lists[1]].view(-1, 4, 4)
        rel_pose = torch.linalg.solve(C1, C2, left=False)
        edge_feats = rel_pose.reshape(-1, 16)
        x = torch.cat([z, poses], dim=-1)
        return Data(x=x, edge_index=adj_lists, edge_attr=edge_feats)

    def simplices_to_adj_list(self, simplices):
        adj_mat = torch.zeros(
            (self.n_vertices, self.n_vertices), dtype=int, device=simplices.device
        )
        x1, x2, x3 = simplices.T

        adj_mat[x1, x2] = 1
        adj_mat[x2, x1] = 1
        adj_mat[x1, x3] = 1
        adj_mat[x3, x1] = 1
        adj_mat[x2, x3] = 1
        adj_mat[x3, x2] = 1
        adj_list = adj_mat.nonzero().t().contiguous()
        return adj_list


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = LookAtMappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img


# ----------------------------------------------------------------------------

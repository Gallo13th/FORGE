import glob
import math
import os
import random
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from Bio import SeqIO
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import kurtosis, skew
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 配置参数


class DataConfig:
    k_neighbors = 32
    coord_fill_null = True
    seq_vocab = "AUCG"
    rbf_dim = 32


def _safe_norm(x, dim=-1, keepdims=False, eps=1e-8):
    return torch.linalg.norm(x, dim=dim, keepdims=keepdims) + eps
    

class BasicStats:
    """
    Unified basic statistical, geometric, and sequence operators for feature engineering.
    """

    @staticmethod
    def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """
        L2 normalize array x along specified dim. Acts as unit_vector.
        """
        return x / (_safe_norm(x, dim=dim, keepdims=True, eps=eps))

    # alias for semantic clarity
    unit_vector = normalize

    @staticmethod
    def distance(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute pairwise Euclidean distance between a and b.
        """
        return _safe_norm(a - b, dim=dim, eps=eps)

    @staticmethod
    def rbf(d: torch.Tensor, num_rbf: int = 32) -> torch.Tensor:
        """
        Radial Basis Function expansion of distance d.
        """
        centers = torch.linspace(0, 30, num_rbf).to(d.device)
        widths = torch.ones(num_rbf).to(d.device) * 1.5
        d_exp = d.unsqueeze(-1)
        return torch.exp(-(((d_exp - centers) / widths) ** 2))

    @staticmethod
    def angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Return angle at b formed by points a-b-c in radians.
        """
        ba = BasicStats.normalize(a - b, dim=-1, eps=eps)
        bc = BasicStats.normalize(c - b, dim=-1, eps=eps)
        cos_theta = torch.clip(torch.sum(ba * bc, dim=-1), -1.0, 1.0)
        return torch.arccos(cos_theta)

    @staticmethod
    def dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute dihedral angle sin and cos between four points p0-p1-p2-p3.
        Returns array of shape (...,2).
        """
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        n1 = torch.cross(b0, b1)
        n2 = torch.cross(b1, b2)
        m1 = torch.cross(n1, BasicStats.normalize(b1, dim=-1, eps=eps))
        x = torch.einsum("...i,...i->...", n1, n2)
        y = torch.einsum("...i,...i->...", m1, n2)
        ang = torch.arctan2(y, x)
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

    @staticmethod
    def plane_normal(pts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute unit normal vector(s) of plane defined by three points.
        pts: (...,3,D) array of points
        Returns: (...,D) normals
        """
        v1 = pts[..., 1, :] - pts[..., 0, :]
        v2 = pts[..., 2, :] - pts[..., 0, :]
        n = torch.cross(v1, v2)
        return BasicStats.normalize(n, dim=-1, eps=eps)

    @staticmethod
    def torsion_rate(coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Approximate torsion τ along a chain of coordinates.
        coords: (L,3)
        Returns: (L,) torsion, with first two and last two elements set to zero.
        """
        L = coords.size(0)
        if L < 4:
            return torch.zeros(L, dtype=coords.dtype, device=coords.device)

        dif1, = torch.gradient(coords, dim=0)
        dif2, = torch.gradient(dif1, dim=0)
        dif3, = torch.gradient(dif2, dim=0)
        num = torch.einsum("ij,ij->i", dif1, torch.cross(dif2, dif3))
        den = (_safe_norm(torch.cross(dif1, dif2), dim=-1) ** 2) + eps
        tau = num / den

        tau = tau.clone()
        tau[:2] = 0.0
        tau[-2:] = 0.0
        return tau

    @staticmethod
    def curvature(coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        L, D = coords.shape
        if L < 3:
            return torch.zeros(L, dtype=coords.dtype, device=coords.device)
        
        d1 = torch.empty((L, D), dtype=coords.dtype, device=coords.device)
        d1[0]   = coords[1]   - coords[0]
        d1[-1]  = coords[-1]  - coords[-2]
        d1[1:-1] = (coords[2:] - coords[:-2]) * 0.5

        d2 = torch.empty((L, D), dtype=coords.dtype, device=coords.device)
        d2[0]   = d1[1]   - d1[0]
        d2[-1]  = d1[-1]  - d1[-2]
        d2[1:-1] = (d1[2:] - d1[:-2]) * 0.5

        cross = torch.cross(d1, d2, dim=-1)                  # (L, D)
        num   = cross.norm(dim=-1)                           # (L,)
        den   = d1.norm(dim=-1).pow(3).add(eps)               # (L,)
        kappa = num.div(den)                                 # (L,)

        return kappa

    @staticmethod
    def frenet_curvature_torsion(P: torch.Tensor, eps: float = 1e-8) -> tuple:
        """
        Compute Frenet curvature and torsion for space curve P (L,3).
        Returns two arrays (L,) for curvature and torsion; endpoints zeroed.
        """
        kappa = BasicStats.curvature(P, eps=eps)
        tau = BasicStats.torsion_rate(P, eps=eps)
        if len(kappa) > 0:
            kappa[0] = kappa[-1] = 0.0
        if len(tau) > 2:
            tau[:2] = tau[-2:] = 0.0
        return kappa, tau

    @staticmethod
    def pair_angle(coord: torch.Tensor, idx1: torch.Tensor, idx2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute angle between coord[idx1]->coord[idx2] and coord[idx2]->coord[idx2+1].
        """
        next_idx = torch.clip(idx2 + 1, 0, coord.shape[0] - 1)
        return BasicStats.angle(coord[idx1], coord[idx2], coord[next_idx], eps=eps)

    @staticmethod
    def _sliding_stat(
        arr: torch.Tensor,
        win: int,
        fn: Callable[[torch.Tensor], torch.Tensor],
        mode: str = "replicate",
    ) -> torch.Tensor:
        """
        一维向量 arr 上的滑窗统计（边界复制填充）
        - arr: (L,) Tensor
        - win: 窗口大小（正奇数更常见）
        - fn: 对形状 (L, win) Tensor 按最后一维计算统计，返回 (L,) Tensor
        - mode: 只支持 "replicate"（复制边缘值）或 "constant"（填 0）
        """
        L = arr.size(0)
        if L == 0 or win <= 1:
            return arr.clone()

        # 1) 计算左右需要填充的长度
        pad_l = win // 2
        pad_r = win - 1 - pad_l

        # 2) 生成填充后的 1D Tensor
        if mode == "replicate":
            left_pad  = arr[:1].expand(pad_l)
            right_pad = arr[-1:].expand(pad_r)
        else:  # constant
            left_pad  = arr.new_zeros(pad_l)
            right_pad = arr.new_zeros(pad_r)

        arr_pad = torch.cat([left_pad, arr, right_pad], dim=0)  # (L + pad_l + pad_r,)
        windows = arr_pad.unfold(0, win, 1)

        return fn(windows)

    @staticmethod
    def window_std(arr: torch.Tensor, win: int) -> torch.Tensor:
        return BasicStats._sliding_stat(
            arr, win,
            fn=lambda w: w.std(dim=1, unbiased=False),
            mode="replicate"
        )

    @staticmethod
    def window_mean(arr: torch.Tensor, win: int) -> torch.Tensor:
        return BasicStats._sliding_stat(
            arr, win,
            fn=lambda w: w.mean(dim=1),
            mode="replicate"
        )

    @staticmethod
    def window_norm_mean(arr: torch.Tensor, win: int) -> torch.Tensor:
        return BasicStats._sliding_stat(
            arr, win,
            fn=lambda w: torch.linalg.norm(w.mean(dim=1), dim=-1),
            mode="replicate"
        )

    @staticmethod
    def window_cv(arr: torch.Tensor, win: int, eps: float = 1e-8) -> torch.Tensor:
        return BasicStats._sliding_stat(
            arr, win,
            fn=lambda w: w.std(dim=1, unbiased=False) / (w.mean(dim=1) + eps),
            mode="replicate"
        )


class RNAEdgeFeatures:
    """
    计算任意 (row,col) 残基对的多尺度几何特征
    coord      : (L,7,3) P,O5',C5',C4',C3',O3',C2'
    edge_index : (2,E)
    返回       : (E,D)  —  默认 D = 32*6 + 3 + 4 + 32 + 40 + 4 + 4 + 2 + 2
    """

    # --------- 1. 基础向量运算 ---------
    @staticmethod
    def _pair_angle(P: torch.Tensor, row: torch.Tensor, col: torch.Tensor, k: int = 1):
        """∠(j→i , i→i+k)"""
        v1 = BasicStats.normalize(P[col] - P[row])
        nxt = torch.clip(col + k, 0, len(P) - 1)
        v2 = BasicStats.normalize(P[nxt] - P[col])
        return BasicStats.angle(v1, torch.zeros_like(v1), -v2)  # (E,)

    @staticmethod
    def _edge_dihedral(P: torch.Tensor, row: torch.Tensor, col: torch.Tensor):
        """四元组 (j‑1, j, i, i+1)"""
        L = len(P)
        a = torch.clip(col - 1, 0, L - 1)
        d = torch.clip(row + 1, 0, L - 1)
        sc = BasicStats.dihedral(P[a], P[col], P[row], P[d])  # (E,2)
        return sc

    @staticmethod
    def _edge_trihedral(P: torch.Tensor, row: torch.Tensor, col: torch.Tensor):
        """三元组 (i, j, j+1) 余弦对映"""
        L = len(P)
        c = torch.clip(row + 1, 0, L - 1)
        v1 = BasicStats.normalize(P[row] - P[col])
        v2 = BasicStats.normalize(P[c] - P[row])
        ang = BasicStats.angle(v1, torch.zeros_like(v1), -v2)
        return torch.stack([torch.sin(ang), torch.cos(ang)], -1)  # (E,2)

    # --------- 2. trRosetta‑style 40D ---------
    @staticmethod
    def _trRosetta(coord: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor]):
        ca, cb = coord[:, 1], coord[:, 6]
        r, c = edge_index
        d = BasicStats.distance(ca[r], ca[c])
        rbf_d = BasicStats.rbf(d)  # (E,32)

        u_ij = BasicStats.normalize(ca[c] - ca[r])
        n_i = BasicStats.normalize(cb[r] - ca[r])
        n_j = BasicStats.normalize(cb[c] - ca[c])

        cos_t1 = torch.clip((u_ij * n_i).sum(-1), -1, 1)
        cos_t2 = torch.clip((-u_ij * n_j).sum(-1), -1, 1)
        theta = torch.stack([torch.sin(torch.arccos(cos_t1)), torch.cos(torch.arccos(cos_t1)), torch.sin(torch.arccos(cos_t2)), torch.cos(torch.arccos(cos_t2))], -1)  # (E,4)

        phi = BasicStats.dihedral(cb[r], ca[r], ca[c], cb[c])  # (E,2)

        L = len(ca)
        omg = BasicStats.dihedral(ca[torch.clip(r - 1, 0, None)], ca[r], ca[c], ca[torch.clip(c + 1, 0, L - 1)])  # (E,2)

        return torch.cat([rbf_d, theta, phi, omg], -1)  # (E,40)

    # --------- 3. 局部取向 (direct+quat) ---------
    @staticmethod
    def _orient_features(coord: torch.Tensor, edge_index):
        P, C5, C4, C3 = coord[:, 0], coord[:, 2], coord[:, 3], coord[:, 4]

        u45 = BasicStats.normalize(C5 - C4)
        u43 = BasicStats.normalize(C3 - C4)
        e1 = BasicStats.normalize(u43 - u45)
        e2 = BasicStats.normalize(torch.cross(u45, u43))
        e3 = torch.cross(e1, e2)
        Q = torch.stack((e1, e2, e3), -2)  # (L,3,3)

        row, col = edge_index
        E_direct = BasicStats.normalize(P[col] - P[row])  # (E,3)

        R_rel = torch.einsum("eji,ejk->eik", Q[row], Q[col])
        tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        w = torch.sqrt(torch.clip(1 + tr, 0, None)) / 2
        xyz = torch.stack([(R_rel[:, 2, 1] - R_rel[:, 1, 2]), (R_rel[:, 0, 2] - R_rel[:, 2, 0]), (R_rel[:, 1, 0] - R_rel[:, 0, 1])], -1) / (4 * w[:, None] + 1e-9)
        E_quat = BasicStats.normalize(torch.cat([xyz, w[:, None]], -1))  # (E,4)
        return E_direct, E_quat

    @staticmethod
    def _cec_pos_enc(delta, n_freq=8, base=200):
        d = torch.sign(delta) * torch.log1p(torch.abs(delta))
        periods = torch.logspace(math.log10(base), math.log10(2), n_freq // 2).to(delta.device)
        beta = 2 * torch.pi / periods
        alpha = beta / torch.pi
        ang = d[:, None] * beta
        ang = torch.remainder(ang, 2 * torch.pi)
        damp = torch.exp(-torch.abs(d)[:, None] * alpha)
        return torch.cat([damp * torch.sin(ang), damp * torch.cos(ang)], -1)

    @staticmethod
    def _stack_flag(coord, row, col):
        N, O3, C4, P = coord[:, 6], coord[:, 5], coord[:, 3], coord[:, 0]
        hb = BasicStats.distance(N[row], O3[col]) < 3.5
        n1 = BasicStats.normalize(torch.cross(BasicStats.normalize(C4 - P), BasicStats.normalize(N - C4)))
        stack = ((n1[row] * n1[col]).sum(-1) > 0.85) & (BasicStats.distance(N[row], N[col]) < 4.0)
        return stack

    # --------- 4. 主入口 ---------
    @staticmethod
    def build(coord: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor]):
        row, col = edge_index

        # 4‑1  多原子对距离 (6×RBF32)
        lookup = {"P": 0, "O5": 1, "C5": 2, "C4": 3, "C3": 4, "O3": 5, "N": 6}
        pairs = [("P", "P"), ("O5", "P"), ("C5", "P"), ("C4", "P"), ("C3", "P"), ("O3", "P"), ("P", "N")]
        dist_blocks = []
        dist_names = []
        for a1, a2 in pairs:
            d = BasicStats.distance(coord[row, lookup[a1]], coord[col, lookup[a2]])
            dist_blocks.append(BasicStats.rbf(d))  # (E,32)
            dist_names.extend([f"dist_{a1}-{a2}_rbf{k}" for k in range(32)])
        E_dist = torch.cat(dist_blocks, -1)  # (E,32*6)

        # 4‑2  Direct / quaternion
        E_direct, E_quat = RNAEdgeFeatures._orient_features(coord, edge_index)  # (E,3),(E,4)
        direct_names = [f"direct_{ax}" for ax in "xyz"]
        quat_names = [f"quat_{i}" for i in range(4)]

        # 4‑3  Relative positional encoding (32D)
        delta = (row - col).float()
        d_model = 32
        E_pe = torch.zeros((len(delta), d_model), dtype=torch.float32, device=delta.device)
        div = torch.exp(torch.arange(0, d_model, 2, device=delta.device) * (-math.log(10000.0) / d_model))
        E_pe[:, 0::2] = torch.sin(delta.unsqueeze(1) * div)
        E_pe[:, 1::2] = torch.cos(delta.unsqueeze(1) * div)
        pe_names = [f"pe_{i}" for i in range(d_model)]

        # CEC Pos Enc
        E_cec = RNAEdgeFeatures._cec_pos_enc(delta, 8)
        cec_names = [f"cec_{i}" for i in range(E_cec.shape[1])]
        
        # 4‑4  trRosetta 40D
        E_rosetta = RNAEdgeFeatures._trRosetta(coord, edge_index)
        rosetta_names = [f"rosetta_{i}" for i in range(E_rosetta.shape[1])]
        
        # 4‑5  pair‑angle / dihedral / trihedral
        pair_ang = torch.stack([RNAEdgeFeatures._pair_angle(coord[:, 0], row, col, k) for k in (1, 2, 3, 4)], -1)  # (E,4)
        pair_ang_names = [f"pair_angle_{k}" for k in range(1, 5)]
        
        dih_blocks, tri_blocks = [], []
        dih_names, tri_names = [], []
        for i in range(coord.shape[1]):
            dih_blocks.append(RNAEdgeFeatures._edge_dihedral(coord[:, i], row, col))  # (E,2)
            tri_blocks.append(RNAEdgeFeatures._edge_trihedral(coord[:, i], row, col))  # (E,2)
            dih_names.extend([f"dih_{i}_{j}" for j in range(2)])
            tri_names.extend([f"tri_{i}_{j}" for j in range(2)])

        dih = torch.cat(dih_blocks, -1)
        tri = torch.cat(tri_blocks, -1)
        stack = ((RNAEdgeFeatures._stack_flag(coord, row, col))).float()[:, None]
        stack_names = ["stack_flag"]
        
        all_features = torch.cat([E_dist, E_direct, E_quat, E_pe, E_rosetta, E_cec, pair_ang, dih, tri, stack], -1)
        all_names = [dist_names,
        direct_names,
        quat_names,
        pe_names,
        rosetta_names,
        cec_names,
        pair_ang_names,
        dih_names,
        tri_names,
        stack_names]
        all_names = sum(all_names, [])
        assert all_features.shape[1] == len(all_names)
        # 4‑6  拼接
        return all_features, all_names


def pad1d_replicate(x: torch.Tensor, pad_left: int, pad_right: int = 0) -> torch.Tensor:
    L = x.shape[0]
    if pad_left > 0:
        left = x[0].unsqueeze(0).expand(pad_left)
    else:
        left = x.new_empty(0)
    if pad_right > 0:
        right = x[-1].unsqueeze(0).expand(pad_right)
    else:
        right = x.new_empty(0)
    return torch.cat([left, x, right], dim=0)

def smooth_1d_unfold(score: torch.Tensor, window: int):
    L = score.size(0)
    if window <= 1 or window > L:
        return score.clone()
    pad = window // 2
    # replicate pad 边界
    left  = score[:pad].flip(0)
    right = score[-pad:].flip(0)
    s_pad = torch.cat([left, score, right], dim=0)        # (L+2*pad,)
    # unfold -> (L, window)
    unf = s_pad.unfold(0, window, 1)                      # (L, window)
    return unf.mean(dim=1)                                # (L,)

def p_distance_stats(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    L = P.size(0)
    if L <= 1:
        return torch.zeros(4, device=P.device, dtype=P.dtype)

    # 1) pairwise 距离矩阵
    dP = torch.cdist(P, P)  # (L,L)

    # 2) 取上三角 (不含对角线) 的掩码
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=P.device), diagonal=1)

    # 3) 扁平提取所有距离
    dvals = dP[mask]  # (L*(L-1)/2,)

    # 4) 一次性计算所有矩统计量
    mean = dvals.mean()
    var  = ((dvals - mean) ** 2).mean()
    std  = var.sqrt()
    m3   = ((dvals - mean) ** 3).mean()
    skew = m3 / (std**3 + eps)
    m4   = ((dvals - mean) ** 4).mean()
    kurt = m4 / (std**4 + eps) - 3.0

    return torch.stack([mean, std, skew, kurt], dim=0)  # (4,)
    
        
def extract_residue_features(coord: torch.Tensor) -> dict:
    """
    Compute residue-level (node) features from atomic coordinates.
    Args:
        coord: (L,7,3) atomic coords per residue
    Returns:
        features: dict[str, torch.Tensor] each (L,)
    """
    L = coord.shape[0]
    center = coord.mean(1)
    feats = {}
    names = ["P", "O5", "C5", "C4", "C3", "O3", "N"]
    P, O5, C5, C4, C3, O3, N = [coord[:, i] for i in range(7)]
    ATOM_IDX = {
        "P": 0,
        "O5": 1,
        "C5": 2,
        "C4": 3,
        "C3": 4,
        "O3": 5,
        "N": 6,
    }

    # 1) backbone bonds & short‑range dists
    for i in range(len(names) - 1):
        a1, a2 = names[i], names[i + 1]
        d = BasicStats.distance(coord[:, i], coord[:, i + 1])
        feats[f"bond_{a1}-{a2}"] = d
        
    for i in range(len(names)):
        for j in range(i + 2, min(i + 4, len(names))):
            a1, a2 = names[i], names[j]
            feats[f"dist_{a1}-{a2}"] = BasicStats.distance(coord[:, i], coord[:, j])

    # P–P distance (shifted)
    DIST_PAIRS = [
        ("P", "P"),  # 原逻辑
        ("P", "O5"),
        ("C4", "C4"),
        ("C4", "P"),
    ]

    ANGLE_TRIPLETS = [
        ("P", "P", "P"),  # 原逻辑
        ("P", "C4", "P"),
        ("C4", "C4", "C4"),
    ]

    SHIFTS = (1, 2, 3)
    for a1, a2 in DIST_PAIRS:
        i1, i2 = ATOM_IDX[a1], ATOM_IDX[a2]
        A, B = coord[:, i1], coord[:, i2]  # (L,3) each
        for k in SHIFTS:
            if L > k:
                d = BasicStats.distance(B[k:], A[:-k])  # (L-k,)
                key = f"dist_{a1}-{a2}_shift{k}"
                # feats[key] = F.pad(d, (k, 0), "replicate")[:L]
                feats[key] = pad1d_replicate(d, pad_left=k, pad_right=0)[:L]
            else:
                feats[f"dist_{a1}-{a2}_shift{k}"] = torch.zeros(L, dtype=coord.dtype, device=coord.device)
                
    for aL, aC, aR in ANGLE_TRIPLETS:
        il, ic, ir = ATOM_IDX[aL], ATOM_IDX[aC], ATOM_IDX[aR]
        Lpt, Cpt, Rpt = coord[:, il], coord[:, ic], coord[:, ir]
        for k in SHIFTS:
            if L > 2 * k:
                ang = BasicStats.angle(Lpt[: -2 * k], Cpt[k:-k], Rpt[2 * k :])  # (L-2k,)
                key = f"angle_{aL}_{aC}_{aR}_span{k}"
                feats[key] = F.pad(ang, (k, k))
            else:
                feats[f"angle_{aL}_{aC}_{aR}_span{k}"] = torch.zeros(L, dtype=coord.dtype, device=coord.device)

    # sugar planarity
    cross = torch.cross(C5 - C4, O5 - C4)
    feats["sugar_planarity"] = torch.abs((cross * (C3 - C4)).sum(-1)) / (torch.linalg.norm(cross, dim=1) + 1e-8)

    # 3) 1st / 2nd diffs
    delta = torch.diff(coord, dim=0, prepend=coord[:1])
    delta2 = torch.diff(delta, dim=0, prepend=delta[:1])
    for atom_idx, a in enumerate(names):
        feats.update({f"delta_{a}_{ax}": delta[:, atom_idx, d] for d, ax in enumerate("xyz")})
        feats.update({f"delta2_{a}_{ax}": delta2[:, atom_idx, d] for d, ax in enumerate("xyz")})

    # 4) residue center
    feats.update({f"center_{ax}": center[:, d] for d, ax in enumerate("xyz")})

    # 5) P‑distance matrix stats # 加到NN里掉分
    if L > 3:
        P = coord[:, RNAGeometry.P]  # (L,3)
        stats = p_distance_stats(P)  # (4,)

        L = P.size(0)
        # 对每个统计量一次 repeat 而非 per-node 循环
        name_fn = ["mean", "std", "skew", "kurtosis"]
        for name, val in zip(name_fn, stats):
            feats[f"p_dist_{name}"] = val.expand(L)

    else:
        for n in ["mean", "std", "skew", "kurtosis"]:
            feats[f"p_dist_{n}"] = torch.zeros(L)

    # 6) curvature using BasicStats.curvature
    for idx, name in enumerate(names):
        kappa = BasicStats.curvature(coord[:, idx])
        feats[f"{name}_kappa_curvature"] = kappa

    # 7) dihedral sin/cos for consecutive P
    if L >= 4:
        dih = BasicStats.dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])
        feats["dihedral_sin"] = F.pad(dih[:, 0], (3, 0))
        feats["dihedral_cos"] = F.pad(dih[:, 1], (3, 0))
    else:
        feats["dihedral_sin"] = torch.zeros(L, dtype=coord.dtype, device=coord.device)
        feats["dihedral_cos"] = torch.zeros(L, dtype=coord.dtype, device=coord.device)

    # 8) sliding‑window std of P
    feats["p_window_std"] = BasicStats.window_std(P.mean(dim=1), 5)

    # 9) SASA‑like density (4 Å)
    if L > 1:
        cm_dist = torch.cdist(center, center)
        feats["sasa_4A"] = (cm_dist < 4.0).sum(1) - 1
    else:
        feats["sasa_4A"] = torch.zeros(L, dtype=coord.dtype, device=coord.device)

    # 10) P→center unit vector
    vec = BasicStats.normalize(center - P)
    for d, ax in enumerate("xyz"):
        feats[f"vec_p_center_{ax}"] = vec[:, d]

    # 11) inertia eigenvalue stats
    centered = coord - center[:, None, :]
    I = torch.einsum("lij,lik->l jk", centered, centered)
    eigs = torch.linalg.eigvalsh(torch.nan_to_num(I))
    feats["inertia_min"] = eigs[:, 0]
    feats["inertia_max"] = eigs[:, -1]
    feats["inertia_ratio"] = eigs[:, 0] / (eigs[:, -1] + 1e-8)

    # 12) dist to global center & main dim
    gctr = torch.nan_to_num(center.mean(0))
    feats["dist_global_ctr"] = BasicStats.distance(center, gctr)
    if L >= 3:
        _, vecs = torch.linalg.eigh(torch.nan_to_num(torch.cov(coord.reshape(-1, 3).T)))
        dim = vecs[:, -1]
        dim_expanded = dim.unsqueeze(0).expand(center.size(0), 3)  # -> (L,3)
        cross_prod = torch.cross(center - gctr, dim_expanded, dim=1)  # (L,3)
        feats["dist_main_dim"] = torch.norm(cross_prod, dim=1)  # (L,)
    else:
        feats["dist_main_dim"] = torch.zeros(L)

    return feats


class RNAGeometry:
    P, O5, C5, C4, C3, O3, C2 = range(7)

    # ============== help function ============= #
    @staticmethod
    def _unit(v, dim=-1):
        return BasicStats.normalize(v, dim=dim)

    @staticmethod
    def _safe_zero(coord):
        return torch.zeros(len(coord), dtype=coord.dtype, device=coord.device)

    @staticmethod
    def _window_stat(x: torch.Tensor, window: int) -> torch.Tensor:
        L, D = x.shape
        if L < window:
            return RNAGeometry._safe_zero(x)

        pad_l = window // 2
        pad_r = window - 1 - pad_l

        # edge replicate pad
        left  = x[0:1].expand(pad_l, D) if pad_l else x.new_empty((0, D))
        right = x[-1:].expand(pad_r, D) if pad_r else x.new_empty((0, D))
        xp = torch.cat([left, x, right], dim=0)  # (L+window-1, D)

        # sliding windows: (L, window, D)
        win = RNAGeometry._sliding_window_view(xp, window)

        # mean over window then norm vector
        mean_vec = win.mean(dim=1)               # (L, D)
        return torch.norm(mean_vec, dim=1)       # (L,)



    @staticmethod
    def _window_cos_similarity(vecs: torch.Tensor, w: int) -> torch.Tensor:
        
        assert vecs.ndim == 2, "vecs must be (L, D)"
        L, D = vecs.shape
        device = vecs.device
        dtype  = vecs.dtype
    
        if L == 0 or w <= 1:
            return torch.ones(L, dtype=dtype, device=device)
    
        pad_l = w // 2
        pad_r = w - 1 - pad_l 
    
        left  = vecs[0:1].expand(pad_l,  D) if pad_l else vecs.new_empty((0, D))
        right = vecs[-1:].expand(pad_r, D) if pad_r else vecs.new_empty((0, D))
    
        v = torch.cat([left, vecs, right], dim=0)            # (L + w - 1, D)
    
        stride0, stride1 = v.stride()       # (step bytes along row, col)
        win_shape  = (L, w, D)
        win_stride = (stride0, stride0, stride1)
        win = v.as_strided(win_shape, win_stride)            # (L, w, D)
    
        dots = torch.bmm(win, win.transpose(1, 2))           # (L, w, w)
        sims = dots.mean(dim=(1, 2))                         # (L,)
    
        return torch.clamp(sims, 0.0, 1.0)

    @staticmethod
    def sequential_turn_span_angle(coord: torch.Tensor) -> torch.Tensor:
        """
        返回每个残基的转角 ∠(P_{i-2}, P_{i-1}, P_i)   shape (L,)
        边界 0,1 用相邻有效值填充。
        """
        L = len(coord)
        if L < 3:
            return RNAGeometry._safe_zero(coord)
        P = coord[:, RNAGeometry.P]
        ang = BasicStats.angle(P[:-2], P[1:-1], P[2:])  # (L‑2,)
        out = torch.zeros(L).to(coord.device)
        out[1:-1] = ang
        out[0] = ang[0]
        out[-1] = ang[-1]
        return out

    @staticmethod
    def puckering_matrix(coord: torch.Tensor) -> Dict[str, torch.Tensor]:
        L = len(coord)
        # C1',C2',C3',C4',O4' ≈ (C2,C4,C3,C5,O5) 索引
        sug = coord[:, [RNAGeometry.C2, RNAGeometry.C4, RNAGeometry.C3, RNAGeometry.C5, RNAGeometry.O5]]
        v = sug - sug.mean(1, keepdims=True)
        phase = 2 * torch.pi * torch.arange(5) / 5
        q = torch.einsum("lmj,m->lj", v.float(), torch.cos(phase).to(coord.device))
        amp = torch.linalg.norm(q[:, 1:], dim=1)
        phi = torch.arctan2(q[:, 2], q[:, 1])
        return {"puckering_0": amp, "puckering_1": phi}

    @staticmethod
    def base_orientation(base_atoms: torch.Tensor, backbone_atoms: torch.Tensor) -> torch.Tensor:
        """碱基‑主链朝向单位向量 (3,)"""
        vec = base_atoms.mean(0) - backbone_atoms.mean(0)
        return BasicStats.normalize(vec)

    @staticmethod
    def helical_parameters_vec(coord: torch.Tensor) -> torch.Tensor:
        """
        向量化计算 helical parameters for all i=1..L-1:
        rise, twist, roll, tilt → 返回 (L,4)，第0行全零。
        """
        L = coord.size(0)
        if L < 2:
            return coord.new_zeros((L, 4))

        # 1) 切片 prev and curr
        prev = coord[:-1]   # (L-1,7,3)
        curr = coord[1:]    # (L-1,7,3)

        # 2) compute normals for each residue: n = normalize( (C4−P) × (C2−C4) )
        v1_prev = prev[:, RNAGeometry.C4] - prev[:, RNAGeometry.P]   # (L-1,3)
        v2_prev = prev[:, RNAGeometry.C2] - prev[:, RNAGeometry.C4]
        prev_norm = torch.cross(v1_prev, v2_prev, dim=1)
        prev_norm = prev_norm / (prev_norm.norm(dim=1, keepdim=True).clamp_min(1e-12))

        v1_curr = curr[:, RNAGeometry.C4] - curr[:, RNAGeometry.P]
        v2_curr = curr[:, RNAGeometry.C2] - curr[:, RNAGeometry.C4]
        curr_norm = torch.cross(v1_curr, v2_curr, dim=1)
        curr_norm = curr_norm / (curr_norm.norm(dim=1, keepdim=True).clamp_min(1e-12))

        # 3) helix axis direction
        helix_dim = (curr[:, RNAGeometry.C4] - prev[:, RNAGeometry.C4])
        helix_dim = helix_dim / helix_dim.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (L-1,3)

        # 4) rise = dot(curr.C4–prev.C4, (prev_norm+curr_norm)/2)
        delta = curr[:, RNAGeometry.C4] - prev[:, RNAGeometry.C4]
        avg_norm = (prev_norm + curr_norm) * 0.5
        rise = torch.sum(delta * avg_norm, dim=1)  # (L-1,)

        # 5) twist = angle between normals in degrees
        cos_twist = torch.clamp((prev_norm * curr_norm).sum(dim=1), -1.0, 1.0)
        twist = torch.acos(cos_twist) * (180.0 / math.pi)  # (L-1,)

        # 6) roll:
        #    n_perp = curr_norm − proj(curr_norm onto helix_dim)
        proj = (curr_norm * helix_dim).sum(dim=1, keepdim=True) * helix_dim
        n_perp = curr_norm - proj
        n_perp = n_perp / n_perp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos_roll = torch.clamp((prev_norm * n_perp).sum(dim=1), -1.0, 1.0)
        roll = torch.acos(cos_roll) * (180.0 / math.pi)       # (L-1,)

        # 7) tilt = asin( clamp(((curr_norm-prev_norm)·helix_dim), -1,1) ) in degrees
        diff_norm = curr_norm - prev_norm
        sin_tilt = torch.clamp((diff_norm * helix_dim).sum(dim=1), -1.0, 1.0)
        tilt = torch.asin(sin_tilt) * (180.0 / math.pi)       # (L-1,)

        # 8) 拼回 (L,4)，第 0 行置零，其它行依次填入
        helical = torch.zeros((L, 4), dtype=coord.dtype, device=coord.device)
        helical[1:, 0] = rise.to(coord.dtype)
        helical[1:, 1] = twist.to(coord.dtype)
        helical[1:, 2] = roll.to(coord.dtype)
        helical[1:, 3] = tilt.to(coord.dtype)

        return helical

    @staticmethod
    def gaussian_curvature(patch: torch.Tensor) -> torch.Tensor:
        patch = torch.nan_to_num(patch)
        _, s, _ = torch.linalg.svd(patch, full_matrices=False)
        return s[-1] / (s[0] + 1e-5)

    # ---------- sliding_window_view 的 PyTorch 等价实现 ----------
    @staticmethod
    def _sliding_window_view(x: torch.Tensor, win_size: int) -> torch.Tensor:
        """
        Mimic np.lib.stride_tricks.sliding_window_view for a 2D tensor x of shape (L, D).
        Returns a view of shape (L - win_size + 1, win_size, D).
        This is a consistent implementation for all RNAGeometry functions.

        Args:
            win_size (int): Window size.

        Returns:
            torch.Tensor: Strided view of shape (L-win_size+1, win_size, D).
        """
        # Ensure 2D input
        assert x.ndim == 2, "_sliding_window_view requires 2D tensor"
        L, D = x.shape
        # Compute strides
        stride0, stride1 = x.stride()
        # Output shape and strides
        out_shape = (L - win_size + 1, win_size, D)
        out_stride = (stride0, stride0, stride1)
        # Return as_strided view
        return x.as_strided(out_shape, out_stride)

    @staticmethod
    def local_C4_curvature(coord: torch.Tensor,
                           window: int = 5) -> torch.Tensor:
        C4 = coord[:, RNAGeometry.C4] # (L,3)
        L = C4.size(0)
        win_size = 2 * window + 1

        if L < win_size:
            return RNAGeometry._safe_zero(coord)

        patches = RNAGeometry._sliding_window_view(C4, win_size)
        means = patches.mean(dim=1, keepdim=True)
        centered = patches - means           # (N, win_size, 3)
        S = torch.linalg.svd(centered, full_matrices=False)[1]  # (L-win_size+1, 3)

        eps = 1e-5
        curv_vals = S[:, -1] / (S[:, 0] + eps)

        return torch.cat([
            torch.zeros(window, dtype=coord.dtype, device=coord.device),
            curv_vals,
            torch.zeros(window, dtype=coord.dtype, device=coord.device),
        ], axis=0)

    @staticmethod
    def hydration_sites(coord: torch.Tensor, probe_radius: float = 1.5) -> torch.Tensor:
        """
        向量化计算每个残基的潜在水合位点数量：
        - 对所有极性原子（P, O5', O3'）成对计算距离
        - 对每个残基的三个极性原子，统计各自半径内的邻居数 < 3 的数量
        """
        L = coord.shape[0]
        # 把三个极性原子顺序堆叠成 (3L,3)
        polar_coords = torch.cat(
            [
                coord[:, RNAGeometry.P],  # (L,3)
                coord[:, RNAGeometry.O5],
                coord[:, RNAGeometry.O3],
            ],
            dim=0,
        )  # 现在形状是 (3L,3)

        # 距离矩阵 (3L,3L)
        D = torch.cdist(polar_coords, polar_coords)
        counts = torch.sum(D < (2 * probe_radius), dim=1)  # (3L,)
        counts = counts.reshape(3, L)
        hydration = torch.sum(counts < 3, dim=0)  # (L,)

        return hydration

    @staticmethod
    def base_stacking(coord: torch.Tensor, window: int = 3) -> torch.Tensor:
        L = coord.size(0)
        normals = torch.cross(coord[:, 6] - coord[:, 3], coord[:, 0] - coord[:, 3], dim=-1)
        normals = BasicStats.normalize(normals)
        center = coord[:, 6]
        score = torch.zeros(L, dtype=coord.dtype, device=coord.device)
        d = BasicStats.distance(center[1:], center[:-1])
        ang = torch.acos(torch.clamp((normals[1:] * normals[:-1]).sum(-1), -1.0, 1.0))
        score[1:] = torch.exp(-d / 3.4) * torch.cos(ang)

        return smooth_1d_unfold(score, window)

    # 1) backbone_direction_vec
    @staticmethod
    def backbone_direction_vec(coord, window=5):
        L = len(coord)
        if L <= 2:
            return RNAGeometry._safe_zero(coord)
            
        v = coord[1:, RNAGeometry.O3] - coord[:-1, RNAGeometry.P]  # (L‑1,3)
        v = RNAGeometry._unit(v)
        sim = RNAGeometry._window_cos_similarity(v, window)  # (L‑1,)
        sim = torch.cat([sim, sim[-1:].expand(1)], dim=0)
        return torch.clamp(sim, 0.0, 1.0)

    # 2) local_electrostatic_clustering_vec
    @staticmethod
    def local_electrostatic_clustering_vec(coord: torch.Tensor,
                                           window: int = 11) -> torch.Tensor:
        """
        使用 cumsum 实现移动平均，取对角线：
        - coord: (L,7,3)
        - window: 滑窗大小
        """

        P = coord[:, RNAGeometry.P]                   # (L,3)
        dist = torch.cdist(P, P)                      # (L,L)
        dist.fill_diagonal_(float('inf'))
        inv = 1.0 / (dist + 1e-8)                     # (L,L)
        pad_l = window // 2
        pad_r = window - 1 - pad_l
        inv_padded = F.pad(inv, (pad_l, pad_r), mode='constant', value=0.0)
        cs = inv_padded.cumsum(dim=1)                 # (L, L+window-1)
        smooth = (cs[:, window - 1:] - cs[:, :-(window - 1)]) / window
        return smooth.diagonal()
        
    # 3) base_stacking_density_vec
    @staticmethod
    def base_stacking_density_vec(coord: torch.Tensor,
                                  window: int = 5) -> torch.Tensor:
        L = coord.size(0)
        if L < 2:
            return RNAGeometry._safe_zero(coord)

        c4   = coord[:, RNAGeometry.C4]                # (L,3)
        base = coord[:, RNAGeometry.C2]                # (L,3)
        
        n = RNAGeometry._unit(base - c4)         # (L,3)
        d = BasicStats.distance(base[1:], base[:-1])
        angs = BasicStats.angle(n[:-1], n[1:], n[1:])
        score = torch.exp(-0.5 * ((d/3.4)**2 + (angs/0.5)**2))  # (L-1,)

        pad = window // 2
        left  = score[0:1].expand(pad)    # (pad,)
        right = score[-1:].expand(pad)    # (pad,)
        score_p = torch.cat([left, score, right], dim=0)  # ((L-1)+2*pad,)

        N = L - 1
        stride = score_p.stride()[0]
        win = score_p.as_strided((N, window), (stride, stride))  # (L-1, window)

        mean = win.mean(dim=1)  # (L-1,)
        out = torch.zeros(L, dtype=coord.dtype, device=coord.device)
        out[1:] = mean[: L-1]

        return out


    # 4) dynamic_betweenness_vec
    @staticmethod
    def dynamic_betweenness_vec(coord: torch.Tensor,
                                window: int = 21,
                                thresh: float = 12.0) -> torch.Tensor:
        # 1) 尺寸和边界
        L = coord.size(0)
        if L < 2:
            return coord.new_zeros(L)

        c4  = coord[:, RNAGeometry.C4]  # (L,3)
        dist = torch.cdist(c4, c4)                               # (L,L)
        adj  = (dist < thresh).to(torch.float64)                 # (L,L)
        adj.fill_diagonal_(0.0)

        p = window // 2
        adj_p = F.pad(
            adj.unsqueeze(0).unsqueeze(0),  # (1,1,L,L)
            (p, p, p, p),                  # pad Wl, Wr, T, B
            mode='constant',
            value=0.0
        ).squeeze(0).squeeze(0)            # back to (L+2p, L+2p)

        tiles = adj_p.unfold(0, window, 1).unfold(1, window, 1)  # (L, L, w, w)

        idx = torch.arange(L)
        subgraphs = tiles[idx, idx]  # (L, w, w)

        paths = torch.linalg.matrix_power(subgraphs, 3)  # (L, w, w)
        betw = paths.sum(dim=(1, 2)) / (window ** 2)      # (L,)
        return betw.to(dtype=coord.dtype, device=coord.device)

    # 5) local_helical_score_vec
    @staticmethod
    def local_helical_score_vec(coord: torch.Tensor, window: int = 5) -> torch.Tensor:
        c4 = coord[:, RNAGeometry.C4]  # (L,3)
        L = c4.size(0)
        pad = window // 2

        # 2) edge padding (replicate first/last rows)
        if pad > 0:
            left  = c4[0:1].expand(pad, 3)    # (pad,3)
            right = c4[-1:].expand(pad, 3)    # (pad,3)
            c4_p = torch.cat([left, c4, right], dim=0)  # (L+2*pad,3)
        else:
            c4_p = c4

        # 3) sliding windows → (L, window, 3)
        patches = RNAGeometry._sliding_window_view(c4_p, window)  # (L,win,3)

        # 4) center each window
        means = patches.mean(dim=1, keepdim=True)       # (L,1,3)
        centered = torch.nan_to_num(patches - means)    # (L,win,3)

        # 5) compute covariance per window: cov[l] = centered[l].T @ centered[l] / window
        cov = torch.einsum('lwi,lwj->lij', centered, centered) / window  # (L,3,3)

        # 6) eigenvalues (ascending) and helical radius r
        eigs = torch.linalg.eigvalsh(cov)               # (L,3)
        r = torch.sqrt(eigs[:, 0] + eigs[:, 1])         # (L,)

        # 7) final score
        return 1.0 / (r + 1e-8)

    # 6) backbone_torsion_variance
    @staticmethod
    def backbone_torsion_variance(coord, window=5):
        P, C4, C3, O3 = coord[:, RNAGeometry.P], coord[:, RNAGeometry.C4], coord[:, RNAGeometry.C3], coord[:, RNAGeometry.O3]
        angle = BasicStats.angle(P - C4, C3 - C4, O3 - C3)
        return BasicStats.window_std(angle, window)

    # 7) sugar_ring_distortion
    @staticmethod
    def sugar_ring_distortion(coord: torch.Tensor) -> torch.Tensor:
        c4 = coord[:, RNAGeometry.C4]    # (L,3)
        c3 = coord[:, RNAGeometry.C3]    # (L,3)
        o3 = coord[:, RNAGeometry.O3]    # (L,3)
    
        normal = RNAGeometry._unit(torch.cross(c3 - c4, o3 - c4), dim=-1)  # (L,3)
        skeleton = coord[:, :6, :]  # (L,6,3)
        disp = skeleton - c4.unsqueeze(1)           # (L,6,3)
        proj = (disp * normal.unsqueeze(1)).sum(dim=2).abs()  # (L,6)
    
        return proj.mean(dim=1)  # (L,)

    # 8) base_orientation
    @staticmethod
    def base_orientation(coord):
        c4, base = coord[:, RNAGeometry.C4], coord[:, RNAGeometry.C2]
        sugar_norm = RNAGeometry._unit(torch.cross(coord[:, RNAGeometry.C3] - c4, coord[:, RNAGeometry.O3] - c4))
        base_norm = RNAGeometry._unit(base - c4)
        return (sugar_norm * base_norm).sum(1)

    # 9) contact_number
    @staticmethod
    def contact_number(center, cutoff=6.0):
        d = torch.cdist(center, center)
        return ((d < cutoff).long() - torch.eye(len(center), dtype=torch.long, device=center.device)).sum(1)

    # 10) local_density (k‑NN)
    @staticmethod
    def local_density(coord: torch.Tensor, k: int = 5) -> torch.Tensor:
        L = coord.size(0)
        if L <= k:
            return torch.zeros(L, dtype=coord.dtype, device=coord.device)

        c4 = coord[:, RNAGeometry.C4]  # (L,3)
        d = torch.cdist(c4, c4)        # (L,L)
        d.fill_diagonal_(float('inf'))
        kth_dist, _ = d.kthvalue(k + 1, dim=1)  # (L,)
        
        return 1.0 / (kth_dist + 1e-8)

    # 11) ss_tendency
    @staticmethod
    def ss_tendency(coord: torch.Tensor, window: int = 3) -> torch.Tensor:
        c4 = coord[:, RNAGeometry.C4]  # (L,3)
        L = c4.size(0)
        if L <= window + 1:
            return torch.zeros(L, dtype=coord.dtype, device=coord.device)

        vec = c4[window:] - c4[:-window]               # (L-window, 3)
        v1 = vec[:-1]                                  # (L-window-1, 3)
        v2 = vec[1:]                                   # (L-window-1, 3)
        dot = (v1 * v2).sum(dim=1)                     # (M,)
        norm = v1.norm(dim=1) * v2.norm(dim=1) + 1e-5
        cos = torch.clamp(dot / norm, -1.0, 1.0)
        ang = torch.acos(cos)                          # (M,) where M = L-window-1

        M = ang.size(0)
        pad = window // 2
        left  = ang[0:1].expand(pad)                   # (pad,)
        right = ang[-1:].expand(pad)                   # (pad,)
        ang_p = torch.cat([left, ang, right], dim=0)   # (M + 2*pad,)

        stride = ang_p.stride()[0]
        win = ang_p.as_strided((M, window), (stride, stride))  # (M, window)

        stds = win.std(dim=1, unbiased=False)          # (M,)

        out = torch.zeros(L, dtype=coord.dtype, device=coord.device)
        start = window + 1
        out[start : start + M] = stds

        return out.to(dtype=coord.dtype, device=coord.device)

    #  ====================== Rolling ===================== #
    @staticmethod
    def pp_direction_consistency(coord, window=7):
        if len(coord) < 2:
            return RNAGeometry._safe_zero(coord)
        v = RNAGeometry._unit(coord[1:, RNAGeometry.P] - coord[:-1, RNAGeometry.P])
        cons = RNAGeometry._window_cos_similarity(v, window)
        out = torch.zeros(len(coord), device=coord.device, dtype=coord.dtype)
        out[:-1] = cons
        out[-1]  = cons[-1]
        
        return out

    @staticmethod
    def ribose_plane_consistency(coord, window=5):
        if len(coord) < 4:
            return RNAGeometry._safe_zero(coord)
        C1, C2, C3 = coord[:, RNAGeometry.C5], coord[:, RNAGeometry.C4], coord[:, RNAGeometry.C3]
        normals = RNAGeometry._unit(torch.cross(C2 - C1, C3 - C1))
        return RNAGeometry._window_cos_similarity(normals, window)

    @staticmethod
    def base_stack_consistency(coord, window=5):
        if len(coord) < 2:
            return RNAGeometry._safe_zero(coord)
        P, C4, N = coord[:, RNAGeometry.P], coord[:, RNAGeometry.C4], coord[:, RNAGeometry.C2]
        normals = RNAGeometry._unit(torch.cross(C4 - P, N - P))
        return RNAGeometry._window_cos_similarity(normals, window)

    @staticmethod
    def torsion_smoothness(phi: torch.Tensor, window: int = 7) -> torch.Tensor:
        """
        phi : (L,)  任意二面角序列（弧度）
        返回   : (L,)  滑窗内 sin/cos 向量均值的范数；越接近 1 越平滑
        """
        if len(phi) < window:
            return RNAGeometry._safe_zero(phi)
        sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)  # (L,2)

        return RNAGeometry._window_stat(sc, window)

    @staticmethod
    def _backbone_dihedral_phi(coord: torch.Tensor) -> torch.Tensor:
        """
        以连续 4 个 P 原子 (i‑3,i‑2,i‑1,i) 计算主链伪二面角 φ_i  (L,)
        边界 0‑2 号残基设为 0。
        """
        P = coord[:, RNAGeometry.P]  # (L,3)
        if len(P) < 4:
            return RNAGeometry._safe_zero(P)
        sincos = BasicStats.dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])  # (L‑3,2)
        phi = torch.zeros(len(P)).to(coord.device)
        phi[3:] = torch.arctan2(sincos[:, 0], sincos[:, 1])  # 还原弧度
        return phi

    @staticmethod
    def torsion_smoothness_backbone(coord: torch.Tensor, window: int = 7) -> torch.Tensor:
        """
        Backbone φ 序列的滑窗平滑度
        """
        phi = RNAGeometry._backbone_dihedral_phi(coord)
        return RNAGeometry.torsion_smoothness(phi, window)

    @staticmethod
    def bond_PO3_cv(coord, window=9):
        if len(coord) < 2:
            return RNAGeometry._safe_zero(coord)
        length = BasicStats.distance(coord[1:, RNAGeometry.O3], coord[:-1, RNAGeometry.P])
        cv_vals = BasicStats.window_cv(length, window)
        out = torch.zeros(len(coord), dtype=coord.dtype, device=coord.device)
        out[:-1] = cv_vals
        out[-1] = cv_vals[-1]

        return out

    @staticmethod
    def backbone_radius_cv(coord, window=11):
        L = len(coord)
        if L < window:
            return RNAGeometry._safe_zero(coord)
        P = coord[:, RNAGeometry.P]
        c = P.mean(0)
        dim = RNAGeometry._unit(torch.linalg.svd(torch.nan_to_num(P - c), full_matrices=False)[2][0])
        proj = P - (P @ dim)[:, None] * dim
        r = torch.linalg.norm(proj, dim=1)
        out = BasicStats.window_cv(r, window)
        return out

    # —— λ2（局部拉普拉斯第二特征） ——
    @staticmethod
    def _knn_indices(pos, k):
        d2 = torch.sum((pos[:, None] - pos[None]) ** 2, dim=-1)
        return torch.argsort(d2, 1)[:, 1 : k + 1]

    @staticmethod
    def lambda2_local(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
        N = pos.shape[0]
        if N < 3:
            return RNAGeometry._safe_zero(pos)

        k_eff = min(k, N - 1)
        idx = RNAGeometry._knn_indices(pos, k_eff).to(pos.device)  # (N, k_eff)

        M = k_eff + 1
        row_idx = torch.arange(N)[:, None].to(pos.device)
        neigh = torch.cat([row_idx, idx], dim=1)  # (N, M)

        sub = pos[neigh]  # (N, M, 3) 或 (N, M, D)
        diff = sub[:, :, None, :] - sub[:, None, :, :]  # (N, M, M, D)
        d2 = torch.sum(diff * diff, dim=-1)  # (N, M, M)

        W = torch.exp(-d2 / 4.0)  # (N, M, M)
        I = torch.arange(M)
        W[:, I, I] = 0.0

        deg = torch.sum(W, dim=-1)  # (N, M)
        L = -W  # (N, M, M)
        L[:, I, I] = deg

        eigs = torch.linalg.eigvalsh(torch.nan_to_num(L))
        lam2 = eigs[:, 1]

        return lam2

    @staticmethod
    def lambda2_std(coord, window=5, k=8):
        lam = RNAGeometry.lambda2_local(coord[:, RNAGeometry.P], k)
        return BasicStats.window_std(lam, window)

    @staticmethod
    def tilt_sigma(coord, window=5):
        if len(coord) < window:
            return RNAGeometry._safe_zero(coord)
        P, C1, base = coord[:, RNAGeometry.P], coord[:, RNAGeometry.C4], coord[:, RNAGeometry.C2]
        u = RNAGeometry._unit(C1 - P)
        v = RNAGeometry._unit(base - C1)
        ang = torch.arccos(torch.clip(torch.abs((u * v).sum(-1)), 0, 1))
        return BasicStats.window_std(ang, window)

    @staticmethod
    def dm_frobenius_consistency(coord: torch.Tensor, window: int = 7) -> torch.Tensor:
        P = coord[:, RNAGeometry.P]  # (L,3)
        L = P.size(0)
        if L < window:
            return RNAGeometry._safe_zero(coord)

        # 2) edge padding
        pad = window // 2
        left  = P[0:1].expand(pad, 3)       # (pad,3)
        right = P[-1:].expand(pad, 3)       # (pad,3)
        Pp = torch.cat([left, P, right], dim=0)  # (L+2*pad,3)

        # 3) sliding window → (L, window, 3)
        win = RNAGeometry._sliding_window_view(Pp, window)  # (L,window,3)

        # 4) pairwise distance matrix per window: (L, window, window)
        #    win[i,:,None,:] - win[i,None,:,:]  → (L,window,window,3)
        diff = win.unsqueeze(2) - win.unsqueeze(1)
        D = diff.norm(dim=-1)  # (L, window, window)

        # 5) zero-mean adjustment per window
        mean = D.mean(dim=(1,2), keepdim=True)  # (L,1,1)
        D0 = D - mean                           # (L,window,window)

        # 6) Frobenius norm squared: flatten last two dims, norm per row, square
        F = D0.reshape(L, -1).norm(dim=1)**2    # (L,)

        # 7) final score
        return -F / 25.0
        
    # ================ Angle ================== #
    @staticmethod
    def _dihedral_angle(p1, p2, p3, p4):
        """
        BasicStats.dihedral → (sin,cos) 转回弧度
        """
        sc = BasicStats.dihedral(p1, p2, p3, p4)  # (...,2)
        return torch.arctan2(sc[..., 0], sc[..., 1])  # (...,)

    @staticmethod
    def _vector_angle(v1, v2):
        return BasicStats.angle(v1, torch.zeros_like(v1), -v2)  # 利用三点夹角公式

    @staticmethod
    def extract_angle_features(coord: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 backbone α–ζ 二面角、若干三面角 / 向量夹角
        返回键带前缀 'dih_' / 'tri_' / 'va_'
        """
        L = len(coord)
        feats: Dict[str, torch.Tensor] = {}

        # ---- 坐标拔出 ----
        names = ["P", "O5", "C5", "C4", "C3", "O3"]
        P, O5, C5, C4, C3, O3, N = (coord[:, i] for i in range(7))
        pts = dict(P=P, O5=O5, C5=C5, C4=C4, C3=C3, O3=O3, N=N)

        # ===== 1) Backbone dihedral α–ζ =====
        if L >= 4:
            dih = {
                "alpha": RNAGeometry._dihedral_angle(torch.roll(O3, 1, 0), P, O5, C5),
                "beta": RNAGeometry._dihedral_angle(P, O5, C5, C4),
                "gamma": RNAGeometry._dihedral_angle(O5, C5, C4, C3),
                "delta": RNAGeometry._dihedral_angle(C5, C4, C3, O3),
                "epsilon": RNAGeometry._dihedral_angle(C4, C3, O3, torch.roll(P, -1, 0)),
                "zeta": RNAGeometry._dihedral_angle(C3, O3, torch.roll(P, -1, 0), torch.roll(O5, -1, 0)),
                "chi": RNAGeometry._dihedral_angle(C4, N, P, C4),
                "eta": RNAGeometry._dihedral_angle(torch.roll(P, -1, 0), C4, P, O3),
                "theta": RNAGeometry._dihedral_angle(C4, P, O3, torch.roll(P, -1, 0)),
            }

        # ===== 2) Trihedral angles =====
        if L >= 3:
            tri = {}
            tri["P_O3prev_O5"] = BasicStats.angle(torch.roll(pts["O3"], 1, 0), pts["P"], pts["O5"])
            tri["P_O3prev_O5"][0] = 0
            tri["P_O5_C5"] = BasicStats.angle(pts["O5"], pts["P"], pts["C5"])
            tri["C4_C5_C3"] = BasicStats.angle(pts["C5"], pts["C4"], pts["C3"])
            tri["C4_C3_O3"] = BasicStats.angle(pts["C3"], pts["C4"], pts["O3"])
            tri["tau_P_C4_next"] = BasicStats.angle(P, C4, torch.roll(P, -1, 0))
            feats.update({f"tri_{k}": v for k, v in tri.items()})

        # ===== 3) Vector‑vector angles =====
        if L >= 2:
            v1 = pts["O5"] - pts["P"]
            v2 = pts["C4"] - pts["C5"]
            va = {"P_O5_vs_C5_C4": RNAGeometry._vector_angle(v1, v2)}
            if "N" in pts:  # base normal vs backbone
                v_base = pts["N"] - pts["C4"]
                v_bb = pts["P"] - pts["C4"]
                va["base_vs_backbone"] = RNAGeometry._vector_angle(v_base, v_bb)
            feats.update({f"va_{k}": v for k, v in va.items()})

        return feats

    @staticmethod
    def extract_rolling_features(coord) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        for w in (3, 5, 7, 11, 13):
            feats[f"pp_dir_cons_{w}"] = RNAGeometry.pp_direction_consistency(coord, w)
            feats[f"ribose_plane_cons_{w}"] = RNAGeometry.ribose_plane_consistency(coord, w)
            feats[f"base_stack_cons_{w}"] = RNAGeometry.base_stack_consistency(coord, w)
            feats[f"bond_PO_cv_{w}"] = RNAGeometry.bond_PO3_cv(coord, w)
            feats[f"radius_cv_{w}"] = RNAGeometry.backbone_radius_cv(coord, w)
            feats[f"lambda2_std_{w}"] = RNAGeometry.lambda2_std(coord, w)
            feats[f"tilt_sigma_{w}"] = RNAGeometry.tilt_sigma(coord, w)
            feats[f"dm_frob_cons_{w}"] = RNAGeometry.dm_frobenius_consistency(coord, w)
            feats[f"tor_smooth_{w}"] = RNAGeometry.torsion_smoothness_backbone(coord, w)

        return feats

    # ── Category A: Base-pair geometry (pure 3D, no sequence info) ──

    @staticmethod
    def base_pair_geometry(coord) -> Dict[str, torch.Tensor]:
        """Per-residue base-pair geometry scores from 3D coordinates only."""
        L = coord.size(0)
        C4 = coord[:, RNAGeometry.C4]
        N_atom = coord[:, 6]
        P = coord[:, RNAGeometry.P]

        gly = RNAGeometry._unit(N_atom - C4)   # glycosidic bond direction
        base_normal = RNAGeometry._unit(torch.cross(C4 - P, N_atom - C4))

        d_c4 = torch.cdist(C4, C4)
        # direction vectors C4_i → C4_j
        dir_ij = C4.unsqueeze(0) - C4.unsqueeze(1)  # (L, L, 3)
        dir_ij = RNAGeometry._unit(dir_ij + 1e-8)

        # glycosidic vectors should point toward each other
        gly_i_fwd = torch.clamp((gly.unsqueeze(1) * dir_ij).sum(-1), 0, 1)
        gly_j_bwd = torch.clamp((gly.unsqueeze(0) * (-dir_ij)).sum(-1), 0, 1)
        convergence = gly_i_fwd * gly_j_bwd

        # base normals should be roughly parallel
        normal_align = torch.abs((base_normal.unsqueeze(1) * base_normal.unsqueeze(0)).sum(-1))

        # C4' distance score (C1'-C1' ~10.5 Å for WC, C4'-C4' similar)
        dist_score = torch.exp(-((d_c4 - 10.5) ** 2) / 6.0)

        # only consider plausible pair distances
        pair_mask = ((d_c4 > 7.0) & (d_c4 < 16.0)).float()

        pair_score = dist_score * convergence * normal_align * pair_mask
        pair_score.fill_diagonal_(0.0)

        best_score, best_partner = pair_score.max(dim=1)
        top3_mean = pair_score.topk(min(3, max(1, L-1)), dim=1)[0].mean(dim=1)
        n_good = (pair_score > 0.5).sum(dim=1).float()
        partner_dist = d_c4[torch.arange(L, device=coord.device), best_partner]

        return {
            "bp_geom_best": best_score,
            "bp_geom_top3": top3_mean,
            "bp_n_good": n_good,
            "bp_partner_dist": partner_dist,
        }

    @staticmethod
    def glycosidic_geometry(coord) -> Dict[str, torch.Tensor]:
        """Glycosidic bond orientation relative to local backbone — purely geometric."""
        L = coord.size(0)
        C4 = coord[:, RNAGeometry.C4]
        N_atom = coord[:, 6]
        P = coord[:, RNAGeometry.P]
        C3 = coord[:, RNAGeometry.C3]

        gly = RNAGeometry._unit(N_atom - C4)

        # backbone direction (P_i → P_i+1)
        if L >= 2:
            bb_tangent = RNAGeometry._unit(torch.roll(P, -1, 0) - P)
            bb_tangent[-1] = bb_tangent[-2]
        else:
            bb_tangent = torch.zeros_like(gly)

        # angle between glycosidic vector and backbone tangent
        cos_gly_bb = torch.clamp((gly * bb_tangent).sum(-1), -1, 1)
        gly_bb_angle = torch.acos(cos_gly_bb)

        # cross product direction (gly × bb_tangent) — indicates base orientation
        gly_cross_bb = torch.cross(gly, bb_tangent)
        gly_cross_norm = torch.norm(gly_cross_bb, dim=-1)

        # angle between glycosidic vector and C4'-P direction (sugar-base)
        c4_p = RNAGeometry._unit(P - C4)
        cos_gly_sugar = torch.clamp((gly * c4_p).sum(-1), -1, 1)
        gly_sugar_angle = torch.acos(cos_gly_sugar)

        # sugar pucker plane normal vs glycosidic vector
        # plane: C4'-C3'-O3'
        sugar_normal = RNAGeometry._unit(torch.cross(C3 - C4, coord[:, RNAGeometry.O3] - C4))
        cos_gly_sugar_plane = (gly * sugar_normal).sum(-1)

        return {
            "gly_bb_angle": gly_bb_angle,
            "gly_cross_norm": gly_cross_norm,
            "gly_sugar_angle": gly_sugar_angle,
            "gly_sugar_plane_cos": cos_gly_sugar_plane,
        }

    @staticmethod
    def base_normal_consistency(coord, windows=(5, 11)) -> Dict[str, torch.Tensor]:
        """Sliding-window consistency of base plane normals."""
        L = coord.size(0)
        C4 = coord[:, RNAGeometry.C4]
        P = coord[:, RNAGeometry.P]
        N_atom = coord[:, 6]

        base_normal = RNAGeometry._unit(torch.cross(C4 - P, N_atom - C4))

        feats = {}
        for w in windows:
            feats[f"base_norm_cons_{w}"] = RNAGeometry._window_cos_similarity(base_normal, w)
        return feats

    @staticmethod
    def extended_sasa(coord, radii=(6.0, 8.0, 12.0)) -> Dict[str, torch.Tensor]:
        """Multi-scale solvent accessible surface area (atom count within radius)."""
        center = coord.mean(dim=1)
        d = torch.cdist(center, center)
        feats = {}
        for r in radii:
            key = f"sasa_{int(r)}A"
            feats[key] = (d < r).sum(dim=1).float() - 1  # exclude self
        return feats

    @staticmethod
    def helical_window_features(coord, windows=(3, 7)) -> Dict[str, torch.Tensor]:
        """Variance of helical parameters in sliding windows."""
        L = coord.size(0)
        helical = RNAGeometry.helical_parameters_vec(coord)  # (L, 4): rise, twist, roll, tilt
        names = ["rise", "twist", "roll", "tilt"]
        feats = {}
        for w in windows:
            for j, name in enumerate(names):
                key = f"helical_{name}_std_{w}"
                feats[key] = BasicStats.window_std(helical[:, j], w)
        return feats

    # ── Graph topology features (from C4' contact graph) ──

    @staticmethod
    def graph_topology_features(coord, contact_cutoff=12.0) -> Dict[str, torch.Tensor]:
        """Betweenness, clustering, eigenvector centrality from C4' contact graph."""
        L = coord.size(0)
        C4 = coord[:, RNAGeometry.C4]
        d = torch.cdist(C4, C4)

        # adjacency: 1 if C4'-C4' < cutoff, 0 otherwise (no self-loops)
        A = (d < contact_cutoff).float()
        A.fill_diagonal_(0.0)

        feats = {}

        # 1) Local clustering coefficient: C_i = 2*e_i / (k_i*(k_i-1))
        degree = A.sum(dim=1)
        # triangles_i = (A^3)_{ii} / 2  (each triangle counted twice per node)
        A2 = A @ A
        A3 = A2 @ A
        triangles = A3.diagonal() / 2.0
        denom = degree * (degree - 1)
        clust_coef = torch.where(denom > 1, 2.0 * triangles / denom, torch.zeros_like(denom))
        feats["graph_clust_coef"] = clust_coef

        # 2) Approximate betweenness via 2- and 3-hop path counts
        # For node i: how many shortest paths pass through i?
        # Approximate with paths of length ≤ 3:
        #   paths through i ≈ sum_{j≠i,k≠i,j<k} (A_ji * A_ik) / (paths between j,k)
        # Simplified: count 2-hop paths where i is the middle
        two_hop = (A.sum(dim=0) * A.sum(dim=1) - A @ A).diagonal()
        # 3-hop paths through i (approximate)
        three_hop = (A2 @ A).diagonal() - A2.diagonal() * A.diagonal()
        btwn_approx = (two_hop + 0.5 * three_hop) / (L + 1e-8)
        feats["graph_btwn_approx"] = btwn_approx

        # 3) Eigenvector centrality (3 power iterations)
        ec = torch.ones(L, dtype=torch.float32, device=coord.device)
        for _ in range(3):
            ec = A @ ec
            ec = ec / (ec.norm() + 1e-8)
        feats["graph_eigen_cent"] = ec

        # 4) Closeness centrality (harmonic mean of distances)
        # Use graph distance (1 / contact distance as weight, then shortest path)
        d_inv = 1.0 / (d + 1e-8)
        d_inv.fill_diagonal_(0.0)
        closeness = d_inv.sum(dim=1) / (L - 1)
        feats["graph_closeness"] = closeness

        # 5) Degree and degree squared (captures hub-ness)
        feats["graph_degree"] = degree
        feats["graph_degree_sq"] = degree ** 2

        # 6) Square clustering: fraction of possible squares that exist
        # (A²)_{ii} is number of 2-paths from i back to i = 2*edges between neighbors
        # ...actually this is just degree, let's do something different
        # Triad participation: how many triangles is this node part of
        feats["graph_triangles"] = triangles

        # 7) Local efficiency: mean of 1/d_ij for neighbors of i
        # For each edge (i,j): count paths of length 2 between i and j
        # then efficiency = (1/k_i) * sum_{j in N(i)} 1/d(i,j|not direct)
        # Approx: mean harmonic distance to neighbors' neighbors
        neighbor_mask = (A > 0).float()
        harmonic_dist = (d_inv * neighbor_mask).sum(dim=1) / (degree + 1e-8)
        feats["graph_local_efficiency"] = harmonic_dist

        return feats

    # ── Multi-scale k-NN features ──

    @staticmethod
    def multiscale_knn_features(coord, ks=(16, 32, 64, 128)) -> Dict[str, torch.Tensor]:
        """Features at multiple k-NN scales, capturing hierarchical spatial context."""
        L = coord.size(0)
        C4 = coord[:, RNAGeometry.C4]
        d = torch.cdist(C4, C4)
        d.fill_diagonal_(float('inf'))

        # Sort distances for each node
        sorted_d, _ = d.sort(dim=1)  # (L, L-1), sorted by distance

        feats = {}
        scale_means = []
        scale_stds = []
        scale_kth = []
        scale_density = []

        for k in ks:
            k_eff = min(k, L - 1)
            knn_d = sorted_d[:, :k_eff]  # (L, k_eff)

            mean_d = knn_d.mean(dim=1)
            std_d = knn_d.std(dim=1, unbiased=False)
            kth_d = knn_d[:, -1]  # distance to k-th neighbor
            density = k_eff / (kth_d ** 3 + 1e-8)

            feats[f"knn{k}_mean_dist"] = mean_d
            feats[f"knn{k}_std_dist"] = std_d
            feats[f"knn{k}_kth_dist"] = kth_d
            feats[f"knn{k}_density"] = density

            scale_means.append(mean_d)
            scale_stds.append(std_d)
            scale_kth.append(kth_d)
            scale_density.append(density)

        # Cross-scale statistics
        all_means = torch.stack(scale_means, dim=0)  # (4, L)
        all_stds = torch.stack(scale_stds, dim=0)
        all_kth = torch.stack(scale_kth, dim=0)
        all_density = torch.stack(scale_density, dim=0)

        feats["knn_scale_mean_mean"] = all_means.mean(dim=0)
        feats["knn_scale_mean_std"] = all_means.std(dim=0, unbiased=False)
        feats["knn_scale_kth_mean"] = all_kth.mean(dim=0)
        feats["knn_scale_kth_std"] = all_kth.std(dim=0, unbiased=False)
        feats["knn_scale_density_mean"] = all_density.mean(dim=0)
        feats["knn_scale_density_std"] = all_density.std(dim=0, unbiased=False)

        # Key ratios: fine-scale vs coarse-scale
        # ratio k16/k128: >1 means locally dense, <1 means globally connected
        feats["knn_ratio_16_128_mean"] = scale_means[0] / (scale_means[-1] + 1e-8)
        feats["knn_ratio_16_128_kth"] = scale_kth[0] / (scale_kth[-1] + 1e-8)
        feats["knn_ratio_16_128_density"] = scale_density[0] / (scale_density[-1] + 1e-8)

        # k16/k32 — early scale change (local clustering signal)
        feats["knn_ratio_16_32_mean"] = scale_means[0] / (scale_means[1] + 1e-8)
        feats["knn_ratio_16_32_kth"] = scale_kth[0] / (scale_kth[1] + 1e-8)

        # Gradient: how density changes with scale (log-log slope)
        # log(density) vs log(k), slope indicates fractal dimension
        log_k = torch.log(torch.tensor(ks, dtype=torch.float32, device=coord.device))
        log_density = torch.log(all_density + 1e-8)  # (4, L)
        n = float(len(ks))
        sx = log_k.sum()
        sx2 = (log_k * log_k).sum()
        # slope_i = (n * sum_k(log_k * log_d_i) - sx * sum_k(log_d_i)) / (n*sx2 - sx^2)
        sxy = (log_k.unsqueeze(-1) * log_density).sum(dim=0)  # (L,)
        sy = log_density.sum(dim=0)  # (L,)
        slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx + 1e-8)
        feats["knn_fractal_dim"] = slope

        return feats

    # All #

    @staticmethod
    def extract_enrich_features(coord) -> Dict[str, torch.Tensor]:
        L = len(coord)
        feats: Dict[str, torch.Tensor] = {}

        # Part 1
        feats.update(RNAGeometry.puckering_matrix(coord))
        
        helical = RNAGeometry.helical_parameters_vec(coord)
        feats.update(
            {
                "helical_rise":  helical[:, 0],
                "helical_twist": helical[:, 1],
                "helical_roll":  helical[:, 2],
                "helical_tilt":  helical[:, 3],
            }
        )

        feats["C4_local_curvature_5"] = RNAGeometry.local_C4_curvature(coord, 5)
        feats["sequential_turn_span_angle"] = RNAGeometry.sequential_turn_span_angle(coord)

        feats.update(
            {
                "stacking": RNAGeometry.base_stacking(coord),
                "hydration": RNAGeometry.hydration_sites(coord),
            }
        )

        # Part 2
        feats["backbone_dir_cons5"] = RNAGeometry.backbone_direction_vec(coord, 5)
        feats["backbone_dir_cons11"] = RNAGeometry.backbone_direction_vec(coord, 11)

        feats["elec_cluster6"] = RNAGeometry.local_electrostatic_clustering_vec(coord, 6)
        feats["elec_cluster15"] = RNAGeometry.local_electrostatic_clustering_vec(coord, 15)

        feats["stack_density5"] = RNAGeometry.base_stacking_density_vec(coord, 5)
        feats["stack_density11"] = RNAGeometry.base_stacking_density_vec(coord, 11)

        feats["dyn_betw10"] = RNAGeometry.dynamic_betweenness_vec(coord, 10)
        feats["dyn_betw21"] = RNAGeometry.dynamic_betweenness_vec(coord, 21)

        feats["local_helix5"] = RNAGeometry.local_helical_score_vec(coord, 5)
        feats["local_helix11"] = RNAGeometry.local_helical_score_vec(coord, 11)

        feats["torsion_var5"] = RNAGeometry.backbone_torsion_variance(coord, 5)
        feats["torsion_var7"] = RNAGeometry.backbone_torsion_variance(coord, 7)

        feats["ring_distort"] = RNAGeometry.sugar_ring_distortion(coord)
        feats["base_orient"] = RNAGeometry.base_orientation(coord)

        # frenet κ/τ
        kP, tP = BasicStats.frenet_curvature_torsion(coord[:, RNAGeometry.P])
        kC4, tC4 = BasicStats.frenet_curvature_torsion(coord[:, RNAGeometry.C4])
        kO5, tO5 = BasicStats.frenet_curvature_torsion(coord[:, RNAGeometry.O5])

        feats.update(
            {
                "P_kappa": kP,
                "P_tau": tP,
                "C4_kappa": kC4,
                "C4_tau": tC4,
                "O5_kappa": kO5,
                "O5_tau": tO5,
            }
        )

        center = coord.mean(1)
        feats["local_density"] = RNAGeometry.local_density(coord)
        feats["contact_num"] = RNAGeometry.contact_number(center)
        feats["ss_tendency"] = RNAGeometry.ss_tendency(coord)

        # ── New features: base-pair geometry, glycosidic orientation,
        #     multi-scale SASA, base normal consistency, helical window stats ──
        feats.update(RNAGeometry.base_pair_geometry(coord))
        feats.update(RNAGeometry.glycosidic_geometry(coord))
        feats.update(RNAGeometry.base_normal_consistency(coord))
        feats.update(RNAGeometry.extended_sasa(coord))
        feats.update(RNAGeometry.helical_window_features(coord))

        # ── Graph topology & multi-scale k-NN features ──
        feats.update(RNAGeometry.graph_topology_features(coord))
        feats.update(RNAGeometry.multiscale_knn_features(coord))

        return feats


class RNAGraphGlobal:
    """
    针对整条 RNA 主链 (P‑原子) 的全局图/几何统计
    所有距离统一走 BasicStats.distance
    """

    # ---------- 内部工具 ----------
    @staticmethod
    def _adjacency(D: torch.Tensor, sigma: float):
        """高斯核邻接矩阵，主对角置 0"""
        A = torch.exp(-(D**2) / sigma**2)
        A.fill_diagonal_(0.0)
        return A

    # ---------- 图谱指标 ----------
    @staticmethod
    def graph_energy(eig: torch.Tensor) -> float:
        return float(torch.abs(eig).sum())

    @staticmethod
    def spectral_radius(eig) -> float:
        return float(eig.max())

    @staticmethod
    def condition_number(D: torch.Tensor, eps: float = 1e-8) -> float:
        s = torch.linalg.svdvals(torch.nan_to_num(D))
        return 0.0 if s[-1] < eps else float(s[0] / s[-1])

    # ---------- 几何粗糙度 ----------
    @staticmethod
    def jaggedness_pp(P: torch.Tensor) -> float:
        """三点二阶差分的均值 (链粗糙度)"""
        if len(P) < 3:
            return 0.0
        l = BasicStats.distance(P[1:], P[:-1])
        dd = torch.abs(l[2:] - 2 * l[1:-1] + l[:-2])
        return float(dd.mean())

    @staticmethod
    def distance_entropy(D_upper: torch.Tensor, nbins: int = 20) -> float:
        hist, _ = torch.histogram(D_upper, bins=nbins, range=(0, 20), density=True)
        hist = torch.clip(hist, 1e-8, 1.0)
        return float(-(hist * torch.log(hist)).sum())

    # ---------- 伪扭转/键长异常 ----------
    @staticmethod
    def pseudo_torsion_acf(angles: torch.Tensor, lag: int = 1) -> float:
        if len(angles) <= lag:
            return 0.0
        return float(torch.mean(angles[:-lag] * angles[lag:]))

    @staticmethod
    def bond_po3_anomaly_prop(lengths: torch.Tensor, mu: float, sigma: float, thresh: float = 2.0) -> float:
        """P‑O3' 键长异常比例(Z‑score > thresh)"""
        if sigma < 1e-6:
            return 0.0
        z = torch.abs((lengths - mu) / sigma)
        return float((z > thresh).mean())

    # ---------- 综合接口 ----------
    @staticmethod
    def global_shape_stats(xyz: torch.Tensor) -> Dict[str, float]:
        """
        xyz : (L,7,3) backbone coords
        返回 dict，仅基于 P‑原子
        """
        P = xyz[:, 0]
        if len(P) < 3:
            return {k: 0.0 for k in ("graph_energy", "spectral_radius", "cond_number", "jaggedness", "dist_entropy")}
        # 距离矩阵
        D = BasicStats.distance(P[:, None, :], P[None, :, :])  # (L,L)
        eig = torch.linalg.eigvalsh(RNAGraphGlobal._adjacency(torch.nan_to_num(D), sigma=4))
        upper = D[torch.triu_indices(len(P), 1)]

        return {
            "graph_energy": RNAGraphGlobal.graph_energy(eig),
            "spectral_radius": RNAGraphGlobal.spectral_radius(eig),
            "cond_number": RNAGraphGlobal.condition_number(D),
            "jaggedness": RNAGraphGlobal.jaggedness_pp(P),
            "dist_entropy": RNAGraphGlobal.distance_entropy(upper),
        }


def build_graph(coord):
    warnings.filterwarnings("ignore")

    # Edge
    if not isinstance(coord, torch.Tensor):
        coord = torch.from_numpy(coord).float()
    else:
        coord = coord.float()
        
    if DataConfig.coord_fill_null:
        coord = torch.nan_to_num(coord)

    edge_index = knn_graph(coord.mean(dim=1), k=DataConfig.k_neighbors, loop=True)
    edge_feats, edge_feat_names = RNAEdgeFeatures.build(coord, edge_index)

    # Node
    node_feats = {}
    node_base_features = extract_residue_features(coord)
    node_enrich_features = RNAGeometry.extract_enrich_features(coord)
    node_angle_features = RNAGeometry.extract_angle_features(coord)
    node_seq_rolling_features = RNAGeometry.extract_rolling_features(coord)
    for feat_dict in (
        node_base_features,
        node_enrich_features,
        node_angle_features,
        node_seq_rolling_features,
    ):
        node_feats.update(feat_dict)

    node_feat_names = list(node_feats.keys())
    node_feats = torch.stack([node_feats[name] for name in node_feat_names], dim=1)

    # Graph
    graph_feats = RNAGraphGlobal.global_shape_stats(coord)

    return {
        "x": node_feats,
        "node_feat_name": node_feat_names,
        "edge_index": edge_index,
        "edge_attr": edge_feats,
        "edge_feat_name": edge_feat_names,
        "graph_attr": graph_feats,
    }
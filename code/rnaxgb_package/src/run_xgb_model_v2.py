import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings

warnings.filterwarnings("ignore")

from warnings import simplefilter

import lightgbm as lgb
import numpy as np
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import coalesce
from tqdm.auto import tqdm


import math
from collections import defaultdict


def normalize(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def rbf(d, num_rbf=32):
    # Simple RBF implementation - adjust as needed
    centers = np.linspace(0, 30, num_rbf)
    widths = np.ones(num_rbf) * 1.5  # Adjust width as needed
    d = np.expand_dims(d, -1)
    return np.exp(-(((d - centers) / widths) ** 2))


def aggregate_features_mean(edge_feats, targets, num_nodes):
    """Aggregate edge features to target nodes (mean only) - optimized version"""
    edge_dim = edge_feats.shape[1]

    # Create degree counts (with min=1 to avoid division by zero)
    degree = np.bincount(targets, minlength=num_nodes).astype(np.float32)
    degree = np.maximum(degree, 1)[:, np.newaxis]  # (num_nodes, 1)

    # Sum features per node using bincount with weights
    # For each feature dimension, we do a weighted sum using bincount
    summed = np.zeros((num_nodes, edge_dim))
    for d in range(edge_dim):
        summed[:, d] = np.bincount(targets, weights=edge_feats[:, d], minlength=num_nodes)

    # Compute mean
    agg_mean = summed / degree

    return agg_mean


def aggregate_features_std(edge_feats, targets, num_nodes):
    """
    Aggregate edge features to target nodes (standard deviation)

    Args:
        edge_feats: (E, D) array of edge features
        targets: (E,) array of target node indices for each edge
        num_nodes: total number of nodes

    Returns:
        agg_std: (num_nodes, D) array of standard deviations per node
    """
    edge_dim = edge_feats.shape[1]

    # First compute the mean using your existing function
    agg_mean = aggregate_features_mean(edge_feats, targets, num_nodes)  # (N, D)

    # Compute squared differences from mean
    squared_diffs = np.zeros_like(edge_feats)
    for d in range(edge_dim):
        squared_diffs[:, d] = (edge_feats[:, d] - agg_mean[targets, d]) ** 2

    # Sum the squared differences
    sum_sq_diffs = np.zeros((num_nodes, edge_dim))
    for d in range(edge_dim):
        sum_sq_diffs[:, d] = np.bincount(targets, weights=squared_diffs[:, d], minlength=num_nodes)

    # Compute variance and std
    degree = np.bincount(targets, minlength=num_nodes).astype(np.float32)
    degree = np.maximum(degree, 1)[:, np.newaxis]  # (num_nodes, 1)
    agg_var = sum_sq_diffs / (degree + 1e-5)
    agg_std = np.sqrt(agg_var)  # (num_nodes, D)

    return agg_std

# ---- Dihedral Angles ---- #


def dihedral_numpy(p0, p1, p2, p3):
    """Compute dihedral angles in radians between vectors (p1-p0, p2-p1) and (p2-p1, p3-p2)"""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)
    y = np.sum(b0xb1_x_b1xb2 * b1, axis=-1) * (1.0 / (np.linalg.norm(b1, axis=-1) + 1e-6))
    x = np.sum(b0xb1 * b1xb2, axis=-1)

    return np.stack([np.sin(np.arctan2(y, x)), np.cos(np.arctan2(y, x))], axis=-1)  # (...,2)


def trRosetta_features(coord, edge_index):
    """
    Compute trRosetta-style geometric features using NumPy.

    Args:
        coord: (L, 7, 3) array of atomic coordinates (P, O5', C5', C4', C3', O3', C2')
        edge_index: (2, E) array of edge indices

    Returns:
        features: (E, 40) array of edge features [RBF(32) + theta(4) + phi(2) + omega(2)]
    """
    # Extract Cα (index 1) and Cβ (index 6) coordinates
    ca, cb = coord[:, 1], coord[:, 6]
    row, col = edge_index

    # ---- Distance Features ---- #
    d = np.linalg.norm(ca[row] - ca[col], axis=-1)  # (E,)
    rbf_d = rbf(d)  # (E, 32)

    # ---- Orientation Features ---- #
    u_ij = (ca[col] - ca[row]) / (np.linalg.norm(ca[col] - ca[row], axis=-1, keepdims=True) + 1e-6)  # (E,3)
    n_i = (cb[row] - ca[row]) / (np.linalg.norm(cb[row] - ca[row], axis=-1, keepdims=True) + 1e-6)  # (E,3)
    n_j = (cb[col] - ca[col]) / (np.linalg.norm(cb[col] - ca[col], axis=-1, keepdims=True) + 1e-6)  # (E,3)

    cos_t1 = np.sum(u_ij * n_i, axis=-1).clip(-1, 1)  # (E,)
    cos_t2 = np.sum(-u_ij * n_j, axis=-1).clip(-1, 1)  # (E,)

    theta = np.stack([np.sin(np.arccos(cos_t1)), np.cos(np.arccos(cos_t1)), np.sin(
        np.arccos(cos_t2)), np.cos(np.arccos(cos_t2))], axis=-1)  # (E,4)

    phi = dihedral_numpy(cb[row], ca[row], ca[col], cb[col])  # (E,2)

    # Omega angles (i-1, i, j, j+1)
    l = ca.shape[0]
    a = np.clip(row - 1, 0, None)
    b = row
    c = col
    d = np.clip(col + 1, 0, l - 1)
    omg = dihedral_numpy(ca[a], ca[b], ca[c], ca[d])  # (E,2)

    return np.concatenate([rbf_d, theta, phi, omg], axis=-1)  # (E,40)


def dihedral(a, b, c, d):
    """Compute dihedral angles between four points in 3D space"""
    b0, b1, b2 = a - b, c - b, d - c
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    y = (np.cross(n1, n2) * b1).sum(-1) / (np.linalg.norm(b1, axis=-1) + 1e-6)
    x = (n1 * n2).sum(-1)
    ang = np.arctan2(y, x)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


def quat_between(R):
    """Return quaternion converting frame j→i"""

    def mat2quat(M):
        t = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
        r = np.sqrt(1 + t)
        q = np.empty((*M.shape[:-2], 4))
        q[..., 0] = (M[..., 2, 1] - M[..., 1, 2]) / (2 * r)
        q[..., 1] = (M[..., 0, 2] - M[..., 2, 0]) / (2 * r)
        q[..., 2] = (M[..., 1, 0] - M[..., 0, 1]) / (2 * r)
        q[..., 3] = 0.5 * r
        return q

    return lambda row, col: mat2quat(np.einsum("...ji,...jk->...ik", R[row], R[col]))


def pair_angle(coord, edge_index, k_shift=1):
    """Angle between vectors j->i and i->i+k_shift"""
    row, col = edge_index
    v1 = (coord[col] - coord[row]) / (np.linalg.norm(coord[col] - coord[row], axis=-1, keepdims=True) + 1e-6)
    k = np.clip(col + k_shift, 0, coord.shape[0] - 1)
    v2 = (coord[k] - coord[col]) / (np.linalg.norm(coord[k] - coord[col], axis=-1, keepdims=True) + 1e-6)
    cos = (v1 * v2).sum(-1).clip(-1 + 1e-6, 1 - 1e-6)
    angle = np.arccos(cos)
    return angle


def edge_dihedral(coord, edge_index):
    """Dihedral angle for quadruplet (i-1, i, j, j+1)"""
    row, col = edge_index
    l = coord.shape[0]
    a = np.clip(col - 1, 0, None)
    b = col
    c = row
    d = np.clip(row + 1, 0, l - 1)
    return dihedral(coord[a], coord[b], coord[c], coord[d])


def edge_trihedral(coord, edge_index):
    """Torsion angle for triplet (i, j, j+1)"""
    row, col = edge_index
    l = coord.shape[0]
    a = col
    b = row
    c = np.clip(row + 1, 0, l - 1)

    v1 = (coord[b] - coord[a]) / (np.linalg.norm(coord[b] - coord[a], axis=-1, keepdims=True) + 1e-6)
    v2 = (coord[c] - coord[b]) / (np.linalg.norm(coord[c] - coord[b], axis=-1, keepdims=True) + 1e-6)
    cos = (v1 * v2).sum(-1).clip(-1 + 1e-6, 1 - 1e-6)
    ang = np.arccos(cos)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


def edge_orient_features(coords, edge_index):
    """
    coords      (L,7,3)
    edge_index  (2,E)   any order
    Returns
    -------
    E_direct (E,3)   : P_i → P_j unit vector
    E_orient (E,4)   : quaternion rotating from residue i frame to j frame
    """
    P, C5, C4, C3 = coords[:, 0, :], coords[:, 2, :], coords[:, 3, :], coords[:, 4, :]

    # Local basis Q_i for each residue
    u45 = normalize(C5 - C4)
    u43 = normalize(C3 - C4)
    e1 = normalize(u43 - u45)
    e2 = normalize(np.cross(u45, u43))
    e3 = np.cross(e1, e2)
    Q = np.stack((e1, e2, e3), axis=-2)  # (L,3,3)

    # Edge endpoints
    row, col = edge_index
    vec = P[col] - P[row]
    E_direct = normalize(vec)  # (E,3)

    # Local rotation R_row.T @ R_col → quaternion
    Q_row = Q[row]
    Q_col = Q[col]
    R_rel = np.einsum("eji,ejk->eik", Q_row, Q_col)  # (E,3,3)

    diag_sum = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    w = np.sqrt(np.clip(1.0 + diag_sum, 0, None)) / 2
    x = (R_rel[:, 2, 1] - R_rel[:, 1, 2]) / (4 * w + 1e-9)
    y = (R_rel[:, 0, 2] - R_rel[:, 2, 0]) / (4 * w + 1e-9)
    z = (R_rel[:, 1, 0] - R_rel[:, 0, 1]) / (4 * w + 1e-9)
    E_orient = normalize(np.stack([x, y, z, w], axis=-1))

    return E_direct, E_orient


def compute_edge_features(coord, edge_index):
    """Compute edge features (E, D_edge)"""
    row, col = edge_index
    select_dicts = {"P": 0, "O5_": 1, "C5_": 2, "C4_": 3, "C3_": 4, "O3_": 5}
    edge_pairs = ["P-P", "O5_-P", "C5_-P", "C4_-P", "C3_-P", "O3_-P"]
    E_dist_list = []

    for pair in edge_pairs:
        a1, a2 = pair.split("-")
        atom1_all = coord[:, select_dicts[a1]]
        atom2_all = coord[:, select_dicts[a2]]
        d = np.linalg.norm(atom1_all[row] - atom2_all[col], axis=-1)
        E_dist_list.append(rbf(d))  # (E, 32)

    E_dist = np.concatenate(E_dist_list, axis=-1)  # (E, 32*6)

    E_direct, E_orient = edge_orient_features(coord, edge_index)
    E_rosetta = trRosetta_features(coord, edge_index)

    E_quat = quat_between(coord)(row, col)
    E_pair = np.concatenate([pair_angle(coord, edge_index, k) for k in range(1, 5)], axis=-1)
    E_dihedral = edge_dihedral(coord, edge_index)
    E_tri = edge_trihedral(coord, edge_index)

    # Positional encoding
    delta = (row - col).astype(np.float32)  # (E,)
    d_model = 32
    pe = np.zeros((delta.shape[0], d_model))
    position = delta[:, np.newaxis]  # (E,1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (16,)

    pe[:, 0::2] = np.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # odd dimensions
    E_pe = pe  # (E,32)

    # Merge all features
    edge_features = np.concatenate(
        [
            E_dist,
            E_direct,
            E_orient,
            E_pe,
            E_rosetta,
            E_quat,
            E_pair,
            E_dihedral,
            E_tri,
        ],
        axis=-1,
    )

    return edge_features  # (E, D_edge)


def calculate_dihedral_angle(p1, p2, p3, p4):
    """Calculate dihedral angle between four points in space"""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize bond vectors
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-6
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-6
    b3 /= np.linalg.norm(b3, axis=-1, keepdims=True) + 1e-6

    # Calculate normals to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize normals
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-6
    n2 /= np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-6

    # Calculate dihedral angle
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(np.cross(n1, b2) * n2, axis=-1)
    return np.arctan2(y, x)


def calculate_trihedral_angle(p1, p2, p3):
    """Calculate angle between three points (p1-p2-p3)"""
    v1 = p1 - p2
    v2 = p3 - p2

    # Normalize vectors
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6
    v2 /= np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6

    # Calculate angle
    dot = np.sum(v1 * v2, axis=-1)
    return np.arccos(np.clip(dot, -1, 1))


def calculate_vector_angle(v1, v2):
    """Calculate angle between two vectors"""
    v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-6)
    dot = np.sum(v1_norm * v2_norm, axis=-1)
    return np.arccos(np.clip(dot, -1, 1))


def extract_all_angles(coord):
    """Main function to extract angular features from RNA coordinates"""
    L = coord.shape[0]
    results = {}

    # Create coordinate dictionary
    backbone_atoms = ["P", "O5", "C5", "C4", "C3", "O3"]
    coords = {name: coord[:, i] for i, name in enumerate(backbone_atoms)}

    # Add N coordinate if available
    if coord.shape[1] > 6:
        coords["N"] = coord[:, 6]

    # ===== Dihedral Angles =====
    dihedrals = {}

    # Standard RNA backbone dihedrals
    if L >= 4:
        # Alpha: O3(i-1)-P-O5-C5
        alpha = calculate_dihedral_angle(np.roll(coords["O3"], 1, axis=0),
                                         coords["P"], coords["O5"], coords["C5"])  # O3_prev
        alpha[0] = np.nan  # First residue has no previous
        dihedrals["alpha"] = alpha

        # Beta: P-O5-C5-C4
        dihedrals["beta"] = calculate_dihedral_angle(coords["P"], coords["O5"], coords["C5"], coords["C4"])

        # Gamma: O5-C5-C4-C3
        dihedrals["gamma"] = calculate_dihedral_angle(coords["O5"], coords["C5"], coords["C4"], coords["C3"])

        # Delta: C5-C4-C3-O3
        dihedrals["delta"] = calculate_dihedral_angle(coords["C5"], coords["C4"], coords["C3"], coords["O3"])

        # Epsilon: C4-C3-O3-P(i+1)
        epsilon = calculate_dihedral_angle(coords["C4"], coords["C3"], coords["O3"],
                                           np.roll(coords["P"], -1, axis=0))  # P_next
        epsilon[-1] = np.nan  # Last residue has no next
        dihedrals["epsilon"] = epsilon

        # Zeta: C3-O3-P(i+1)-O5(i+1)
        zeta = calculate_dihedral_angle(coords["C3"], coords["O3"], np.roll(
            coords["P"], -1, axis=0), np.roll(coords["O5"], -1, axis=0))  # P_next  # O5_next
        zeta[-1] = np.nan  # Last residue has no next
        dihedrals["zeta"] = zeta

    results.update({"dih_" + k: v for k, v in dihedrals.items()})

    # ===== Trihedral Angles =====
    trihedrals = {}

    if L >= 3:
        # P-centered angles
        trihedrals["P_O3_prev_O5"] = calculate_trihedral_angle(
            np.roll(coords["O3"], 1, axis=0), coords["P"], coords["O5"])  # O3_prev
        trihedrals["P_O3_prev_O5"][0] = np.nan  # First residue
        trihedrals["P_O5_C5"] = calculate_trihedral_angle(coords["O5"], coords["P"], coords["C5"])
        trihedrals["C4_C5_C3"] = calculate_trihedral_angle(coords["C5"], coords["C4"], coords["C3"])
        trihedrals["C4_C3_O3"] = calculate_trihedral_angle(coords["C3"], coords["C4"], coords["O3"])

    results.update({"tri_" + k: v for k, v in trihedrals.items()})

    # ===== Vector Angles =====
    vector_angles = {}

    if L >= 2:
        # P-O5 vs C5-C4
        v1 = coords["O5"] - coords["P"]
        v2 = coords["C4"] - coords["C5"]
        vector_angles["P_O5_vs_C5_C4"] = calculate_vector_angle(v1, v2)

        # Base normal vs backbone (if N exists)
        if "N" in coords:
            v_base = coords["N"] - coords["C4"]
            v_backbone = coords["P"] - coords["C4"]
            vector_angles["base_vs_backbone"] = calculate_vector_angle(v_base, v_backbone)

    results.update({"va_" + k: v for k, v in vector_angles.items()})

    return results


import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import kurtosis, skew


def extract_statistical_features(coord, edge_index):
    """
    提取(L,7,3)坐标的统计特征（所有特征严格对齐到长度L）
    返回: 字典形式的各种统计特征，每个值的形状都是(L,)或标量
    """
    L = coord.shape[0]
    features = {}

    # 原子坐标 (L,3)
    P = coord[:, 0]
    O5 = coord[:, 1]
    C5 = coord[:, 2]
    C4 = coord[:, 3]
    C3 = coord[:, 4]
    O3 = coord[:, 5]
    N = coord[:, 6] if coord.shape[1] > 6 else np.zeros_like(P)

    # ===== 1. 原子间距离特征（全部对齐到L）=====
    # # 骨架键长（边界用最近值填充）
    backbone_atoms = ["P", "O5", "C5", "C4", "C3", "O3", "N"]
    atom_coords = {name: coord[:, i] for i, name in enumerate(backbone_atoms)}
    for i in range(len(backbone_atoms) - 1):
        a1, a2 = backbone_atoms[i], backbone_atoms[i + 1]
        features[f"bond_{a1}-{a2}"] = np.linalg.norm(atom_coords[a1] - atom_coords[a2], axis=1)

    for i in range(len(backbone_atoms)):
        for j in range(i + 1, min(i + 4, len(backbone_atoms))):  # Look ahead up to 3 atoms
            if j == i + 1:
                continue  # Skip consecutive pairs already calculated
            a1, a2 = backbone_atoms[i], backbone_atoms[j]
            dist = np.linalg.norm(atom_coords[a1] - atom_coords[a2], axis=1)
            features[f"dist_{a1}-{a2}"] = dist

    # P-P距离（L-1 -> L）
    p_p_dist = np.linalg.norm(P[1:] - P[:-1], axis=1)
    features["bond_P-P"] = np.concatenate([p_p_dist[:1], p_p_dist])[:L]  # 首元素重复

    # ===== 2. 角度特征（边界填充0）=====
    # P-P-P角度 (L-2 -> L)
    if L >= 3:
        v1 = P[:-2] - P[1:-1]
        v2 = P[2:] - P[1:-1]
        cos_theta = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        angle = np.arccos(np.clip(cos_theta, -1, 1))
        features["angle_P-P-P"] = np.concatenate([[0], angle, [0]])[:L]
    else:
        features["angle_P-P-P"] = np.zeros(L)

    # 糖环平面性 (L,)
    vec1 = C5 - C4
    vec2 = O5 - C4
    vec3 = C3 - C4
    cross = np.cross(vec1, vec2)
    features["sugar_planarity"] = np.abs(np.sum(cross * vec3, axis=1)) / (np.linalg.norm(cross, axis=1) + 1e-6)

    # ===== 3. 差分特征（边界填充0）=====
    # 一阶差分 (L-1 -> L)
    delta_coord = np.diff(coord, axis=0)
    for i, atom in enumerate(["P", "O5", "C5", "C4", "C3", "O3", "N"]):
        for dim, dim_name in enumerate(["x", "y", "z"]):
            key = f"delta_{atom}_{dim_name}"
            features[key] = np.concatenate([delta_coord[:1, i, dim], delta_coord[:, i, dim]])[:L]

    # 二阶差分 (L-2 -> L)
    if L >= 2:
        delta2_coord = np.diff(delta_coord, axis=0)
        for i, atom in enumerate(["P", "O5", "C5", "C4", "C3", "O3", "N"]):
            for dim, dim_name in enumerate(["x", "y", "z"]):
                key = f"delta2_{atom}_{dim_name}"
                features[key] = np.concatenate([[0], delta2_coord[:, i, dim], [0]])[:L]
    else:
        for atom in ["P", "O5", "C5", "C4", "C3", "O3", "N"]:
            for dim in ["x", "y", "z"]:
                features[f"delta2_{atom}_{dim}"] = np.zeros(L)

    # ===== 4. 几何特征（全部L长度）=====
    center = np.mean(coord, axis=1)
    features["center_x"] = center[:, 0]
    features["center_y"] = center[:, 1]
    features["center_z"] = center[:, 2]

    # ===== 5. 距离矩阵特征（全局标量）=====
    p_dist_mat = squareform(pdist(P)) if L > 1 else np.zeros((1, 1))
    triu_values = p_dist_mat[np.triu_indices_from(p_dist_mat, k=1)]
    features["p_dist_mean"] = np.full(L, triu_values.mean() if len(triu_values) > 0 else 0)
    features["p_dist_std"] = np.full(L, triu_values.std() if len(triu_values) > 0 else 0)
    features["p_dist_skew"] = np.full(L, skew(triu_values) if len(triu_values) > 0 else 0)
    features["p_dist_kurtosis"] = np.full(L, kurtosis(triu_values) if len(triu_values) > 0 else 0)

    # ===== 6. 曲率特征（边界填充0）=====
    if L >= 3:
        t1 = P[:-2] - P[1:-1]
        t2 = P[2:] - P[1:-1]
        cross = np.linalg.norm(np.cross(t1, t2), axis=1)
        dot = np.sum(t1 * t2, axis=1)
        curvature = cross / (np.linalg.norm(t1, axis=1) * np.linalg.norm(t2, axis=1) *
                             (np.linalg.norm(t1, axis=1) + np.linalg.norm(t2, axis=1)) + 1e-6)
        features["curvature"] = np.concatenate([[0], curvature, [0]])[:L]
    else:
        features["curvature"] = np.zeros(L)

    if L >= 3:
        # 基于C4'坐标计算离散曲率
        curvature = np.zeros(L)
        for i in range(1, L - 1):
            v1 = C4[i] - C4[i - 1]
            v2 = C4[i + 1] - C4[i]
            curvature[i] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        # 边界填充
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        features["c4_curvature"] = curvature  # (L,)
    else:
        features["c4_curvature"] = np.zeros(L)

    # ===== 7. 二面角特征（边界填充0）=====
    if L >= 4:
        v1 = P[:-3] - P[1:-2]
        v2 = P[2:-1] - P[1:-2]
        v3 = P[3:] - P[2:-1]
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        x = np.sum(n1 * n2, axis=1)
        y = np.sum(np.cross(n1, n2) * v2, axis=1) / (np.linalg.norm(v2, axis=1) + 1e-6)
        dihedral = np.arctan2(y, x)
        features["dihedral_sin"] = np.concatenate([[0, 0, 0], np.sin(dihedral)])[:L]
        features["dihedral_cos"] = np.concatenate([[0, 0, 0], np.cos(dihedral)])[:L]
    else:
        features["dihedral_sin"] = np.zeros(L)
        features["dihedral_cos"] = np.zeros(L)

    # ===== 8. 滑动窗口特征（边界用最近值填充）=====
    window_size = min(5, L)
    if window_size > 1:
        p_std = np.zeros(L)
        for i in range(L):
            start = max(0, i - window_size // 2)
            end = min(L, i + window_size // 2 + 1)
            window = P[start:end]
            p_std[i] = np.std(window, axis=0).mean()
        features["p_window_std"] = p_std
    else:
        features["p_window_std"] = np.zeros(L)

    # ========== 8. 溶剂可及性特征 ==========
    # 简化的SASA近似 (使用4Å半径内的原子计数)
    if L > 1:
        dist_mat = cdist(center, center)
        features["sasa_4A"] = np.sum(dist_mat < 4.0, axis=1) - 1  # 减去自身
    else:
        features["sasa_4A"] = np.zeros(L)

    # ========== 9. 向量场特征 ==========
    # 磷酸指向几何中心的单位向量 (L,3)
    vec_p_center = center - P
    norm = np.linalg.norm(vec_p_center, axis=1) + 1e-6
    features["vec_p_center_x"] = vec_p_center[:, 0] / norm
    features["vec_p_center_y"] = vec_p_center[:, 1] / norm
    features["vec_p_center_z"] = vec_p_center[:, 2] / norm

    # ========== 10. 高阶统计特征 ==========
    # 每个残基的惯性矩 (L,)
    inertia = []
    for i in range(L):
        centered = coord[i] - center[i]
        I = centered.T @ centered  # (3,3)
        eigvals = np.linalg.eigvalsh(I)
        inertia.append(eigvals)
    inertia = np.array(inertia)
    features["inertia_min"] = inertia.min(axis=1)
    features["inertia_max"] = inertia.max(axis=1)
    features["inertia_ratio"] = features["inertia_min"] / (features["inertia_max"] + 1e-6)

    # ========== 11. 滑动窗口统计特征 ==========
    window_size = 5
    if L >= window_size:
        # 滑动窗口内的P原子位置标准差
        p_std = np.zeros(L)
        for i in range(L - window_size + 1):
            window = P[i: i + window_size]
            p_std[i: i + window_size] += np.std(window, axis=0).mean()
        p_std /= L - window_size + 1
        features["p_window_std"] = p_std
    else:
        features["p_window_std"] = np.nan

    # ========== 12. 相对全局特征 ==========
    # 每个残基到全局中心的距离
    global_center = np.mean(coord, axis=(0, 1))
    features["dist_to_global_center"] = np.linalg.norm(center - global_center, axis=1)

    # 每个残基到全局主轴的距离
    if L >= 3:
        _, eigvecs = np.linalg.eigh(np.cov(coord.reshape(-1, 3).T))
        main_axis = eigvecs[:, -1]  # 最大特征值对应的主轴
        features["dist_to_main_axis"] = np.linalg.norm(np.cross(center - global_center, main_axis), axis=1)
    else:
        features["dist_to_main_axis"] = np.zeros(L)

    angle_feats = extract_all_angles(coord)
    features.update(angle_feats)

    # =========== Add ===============
    features.update(RNAGeometry.extract_all(coord))

    P_kappa, P_torsion = frenet_curvature_torsion(P)
    C4_kappa, C4_torsion = frenet_curvature_torsion(C4)
    O5_kappa, O5_torsion = frenet_curvature_torsion(O5)

    features.update(
        {
            "local_density": local_density(coord),
            "ss_tendency": ss_tendency(coord),
            "contact_num": contact_number(center),
            "P_kappa": P_kappa,
            "P_torsion": P_torsion,
            "C4_kappa": C4_kappa,
            "C4_torsion": C4_torsion,
            "O5_kappa": O5_kappa,
            "O5_torsion": O5_torsion,
            "sequential_turn_span_angle": sequential_turn_span_angle(coord),
            "sugar_ring_distortion": sugar_ring_distortion(coord),
            "base_orientation": base_orientation(coord),
            "backbone_torsion_variance": backbone_torsion_variance(coord),
            "local_helical_score_vec_window_5": local_helical_score_vec(coord, 5),
            "local_helical_score_vec_window_11": local_helical_score_vec(coord, 11),
            "base_stacking_density_vec_window_5": base_stacking_density_vec(coord, 5),
            "base_stacking_density_vec_window_11": base_stacking_density_vec(coord, 11),
            "dynamic_betweenness_vec_window_10": dynamic_betweenness_vec(coord, 10),
            "dynamic_betweenness_vec_window_21": dynamic_betweenness_vec(coord, 21),
            "local_electrostatic_clustering_vec_window_6": local_electrostatic_clustering_vec(coord, 6),
            "local_electrostatic_clustering_vec_window_15": local_electrostatic_clustering_vec(coord, 15),
            "backbone_direction_vec_window_5": backbone_direction_vec(coord, 5),
            "backbone_direction_vec_window_11": backbone_direction_vec(coord, 11),
        }
    )

    features.update(all_consistency_features(coord))
    features.update(global_shape_stats(coord))

    # # ===== Edge Feature ======
    # edge_feats_agg = []
    edge_feats = compute_edge_features(coord, edge_index)
    edge_feats_agg = np.concatenate(
        (
            aggregate_features_mean(edge_feats, edge_index[1], coord.shape[0]),
            aggregate_features_std(edge_feats, edge_index[1], coord.shape[0]),
        ), axis=-1
    )

    return features, edge_feats_agg


def dm_graph_energy(D: np.ndarray, sigma: float = 4.0) -> float:
    """图能量; D 方阵"""
    A = np.exp(-D ** 2 / sigma ** 2)
    np.fill_diagonal(A, 0.0)
    eig = np.linalg.eigvalsh(A)
    return float(np.sum(np.abs(eig)))


def dm_spectral_radius(D: np.ndarray, sigma: float = 4.0) -> float:
    A = np.exp(-D ** 2 / sigma ** 2)
    np.fill_diagonal(A, 0.0)
    eig = np.linalg.eigvalsh(A)
    return float(np.max(eig))


def dm_condition_number(D: np.ndarray, eps: float = 1e-8) -> float:
    s = np.linalg.svd(D, compute_uv=False)
    if s[-1] < eps:
        return 0.0
    return float(s[0] / s[-1])


def jaggedness_pp(P: np.ndarray) -> float:
    if len(P) < 3:
        return 0.0
    l = np.linalg.norm(P[1:] - P[:-1], axis=1)
    dd = np.abs(l[2:] - 2 * l[1:-1] + l[:-2])
    return float(dd.mean())


def distance_entropy(D_upper: np.ndarray, nbins: int = 20) -> float:
    hist, _ = np.histogram(D_upper, bins=nbins, range=(0, 20), density=True)
    hist = np.clip(hist, 1e-8, 1.0)
    return float(-(hist * np.log(hist)).sum())


def pseudo_torsion_acf(angles: np.ndarray, lag: int = 1) -> float:
    if len(angles) <= lag:
        return 0.0
    return float(np.mean(angles[:-lag] * angles[lag:]))


def bond_po3_anomaly_prop(lengths: np.ndarray, mu: float, sigma: float, thresh: float = 2.0) -> float:
    if sigma < 1e-6:
        return 0.0
    z = np.abs((lengths - mu) / sigma)
    return float((z > thresh).mean())


def global_shape_stats(xyz: np.ndarray) -> dict:
    """返回全链统计指标"""
    P = xyz[:, 0]
    if len(P) < 3:
        return {
            "graph_energy": 0.0,
            "spectral_radius": 0.0,
            "cond_number": 0.0,
            "jaggedness": 0.0,
            "dist_entropy": 0.0,
        }
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    upper = D[np.triu_indices(len(P), 1)]
    return {
        "graph_energy": dm_graph_energy(D),
        "spectral_radius": dm_spectral_radius(D),
        "cond_number": dm_condition_number(D),
        "jaggedness": jaggedness_pp(P),
        "dist_entropy": distance_entropy(upper),
    }


import numpy as np
from typing import Tuple

# ---------- 基础向量工具 ----------


def unit(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """按 axis 归一化向量"""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def plane_normal(pts: np.ndarray) -> np.ndarray:
    """pts: (...,4,3) 返回拟合平面法向 (...,3)"""
    v1 = pts[..., 1, :] - pts[..., 0, :]
    v2 = pts[..., 2, :] - pts[..., 0, :]
    n = np.cross(v1, v2)
    return unit(n)

# ---------- 通用滑窗统计 ----------


def _safe_zero(L: int) -> np.ndarray:
    """返回全零数组 helper"""
    return np.zeros(L, dtype=float)


def window_cos_similarity(vecs: np.ndarray, w: int) -> np.ndarray:
    L = len(vecs)
    if L == 0:
        return np.empty(0)
    if L == 1 or L < w:
        return np.ones(L)
    out = np.zeros(L)
    half = w // 2
    for i in range(L):
        l = max(0, i - half)
        r = min(L - 1, i + half)
        sub = vecs[l: r + 1]
        if len(sub) == 1:
            out[i] = 1.0
        else:
            dots = sub @ sub.T
            m = len(sub)
            out[i] = (dots.sum() - m) / (m * (m - 1))
    return out


def window_std(arr: np.ndarray, w: int) -> np.ndarray:
    L = len(arr)
    if L == 0:
        return np.empty(0)
    if L < w:
        return _safe_zero(L)
    out = np.zeros(L)
    half = w // 2
    for i in range(L):
        l = max(0, i - half)
        r = min(L - 1, i + half)
        sub = arr[l: r + 1]
        out[i] = np.std(sub)
    return out


def window_cv(arr: np.ndarray, w: int, eps: float = 1e-8) -> np.ndarray:
    L = len(arr)
    if L == 0:
        return np.empty(0)
    if L < w:
        return _safe_zero(L)
    out = np.zeros(L)
    half = w // 2
    for i in range(L):
        l = max(0, i - half)
        r = min(L - 1, i + half)
        sub = arr[l: r + 1]
        mean = np.mean(sub)
        out[i] = np.std(sub) / (mean + eps)
    return out

# ---------- 特征 1: P->P 方向一致性 ----------


def pp_direction_consistency(xyz: np.ndarray, window: int = 7) -> np.ndarray:
    L = len(xyz)
    if L < 2:
        return _safe_zero(L)
    P = xyz[:, 0]
    vecs = unit(P[1:] - P[:-1])
    cons = window_cos_similarity(vecs, window)
    out = np.zeros(L)
    if len(cons):
        out[:-1] = cons
        out[-1] = cons[-1]
    return out

# ---------- 特征 2: 糖环法向一致性 ----------


def ribose_plane_consistency(xyz: np.ndarray, window: int = 5) -> np.ndarray:
    L = len(xyz)
    if L < 4:
        return _safe_zero(L)
    # 使用近似索引：1:O5',2:C5',3:C4',4:C3'
    C1 = xyz[:, 2]
    C2 = xyz[:, 3]
    C3 = xyz[:, 4]
    C4 = xyz[:, 3]  # 占位，同 C2
    pts = np.stack([C1, C2, C3, C4], axis=1)
    normals = plane_normal(pts)
    return window_cos_similarity(normals, window)

# ---------- 特征 3: 基底堆叠一致性 ----------


def base_stack_consistency(xyz: np.ndarray, window: int = 5) -> np.ndarray:
    L = len(xyz)
    if L < 2:
        return _safe_zero(L)
    P = xyz[:, 0]
    C4 = xyz[:, 3]
    N = xyz[:, 6]
    pts = np.stack([P, C4, N, N], axis=1)
    normals = plane_normal(pts)
    return window_cos_similarity(normals, window)

# ---------- 特征 4: 二面角平稳度 ----------


def torsion_smoothness(phi: np.ndarray, window: int = 7) -> np.ndarray:
    L = len(phi)
    if L < window:
        return _safe_zero(L)
    sc = np.stack([np.sin(phi), np.cos(phi)], axis=-1)
    half = window // 2
    out = np.zeros(L)
    for i in range(L):
        l = max(0, i - half)
        r = min(L - 1, i + half)
        sub = sc[l: r + 1]
        mean_vec = np.mean(sub, axis=0)
        out[i] = np.linalg.norm(mean_vec)
    return out

# ---------- 特征 5: 键长 CV ----------


def bond_PO3_cv(xyz: np.ndarray, window: int = 9) -> np.ndarray:
    L = len(xyz)
    if L < 2:
        return _safe_zero(L)
    length = np.linalg.norm(xyz[1:, 5] - xyz[:-1, 0], axis=1)
    cv = window_cv(length, window)
    out = np.zeros(L)
    if len(cv):
        out[:-1] = cv
        out[-1] = cv[-1]
    return out

# ---------- 特征 6: 半径 CV ----------


def backbone_radius_cv(xyz: np.ndarray, window: int = 11) -> np.ndarray:
    L = len(xyz)
    if L < window:
        return _safe_zero(L)
    P = xyz[:, 0]
    c = P.mean(0)
    u = unit(np.linalg.svd(P - c, full_matrices=False)[2][0])
    diff = P - c
    proj = diff - (diff @ u)[:, None] * u
    r = np.linalg.norm(proj, axis=1)
    return window_cv(r, window)

# ---------- 特征 7: 简易 λ2 及其窗口 σ ----------


def _knn_indices(pos: np.ndarray, k: int) -> np.ndarray:
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    return np.argsort(d2, axis=1)[:, 1: k + 1]


def lambda2_local(pos: np.ndarray, k: int = 8) -> np.ndarray:
    N = len(pos)
    if N < 3:
        return _safe_zero(N)
    idx = _knn_indices(pos, min(k, N - 1))
    lam = np.zeros(N)
    for i in range(N):
        neighbors = np.concatenate([[i], idx[i]])
        sub = pos[neighbors]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        w = np.exp(-d2 / 4.0)
        np.fill_diagonal(w, 0.0)
        deg = np.sum(w, axis=1)
        Lmat = np.diag(deg) - w
        eig = np.linalg.eigvalsh(Lmat)
        lam[i] = eig[1] if len(eig) > 1 else 0.0
    return lam


def lambda2_std(xyz: np.ndarray, window: int = 5, k: int = 8) -> np.ndarray:
    L = len(xyz)
    if L < k + 1:
        return _safe_zero(L)
    lam = lambda2_local(xyz[:, 0], k)
    return window_std(lam, window)

# ---------- 特征 8: P-Base 倾角 σ ----------


def tilt_sigma(xyz: np.ndarray, window: int = 5) -> np.ndarray:
    L = len(xyz)
    if L < window:
        return _safe_zero(L)
    P = xyz[:, 0]
    C1 = xyz[:, 3]
    base = xyz[:, 6]
    u = unit(C1 - P)
    v = unit(base - C1)
    cos = np.abs((u * v).sum(-1))
    angles = np.arccos(np.clip(cos, 0, 1))
    return window_std(angles, window)

# ---------- 特征 9: 窗口 Frobenius 一致性 ----------


def dm_frobenius_consistency(xyz: np.ndarray, window: int = 7) -> np.ndarray:
    P = xyz[:, 0]
    L = len(P)
    if L < window:
        return _safe_zero(L)

    half = window // 2
    out = np.zeros(L)
    for i in range(L):
        l = max(0, i - half)
        r = min(L - 1, i + half)
        sub = P[l: r + 1]
        D = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        D0 = D - D.mean()
        # out[i] = np.exp(-np.square(np.linalg.norm(D0)) / 25.0)
        out[i] = -np.square(np.linalg.norm(D0)) / 25.0
    return out

# ---------- 公开统一接口 ----------


def all_consistency_features(xyz: np.ndarray) -> dict:
    """返回一个字典，键=特征名, 值=(L,) ndarray"""
    feats = {}
    for window in [3, 5, 7, 11, 13]:
        feats[f"pp_dir_cons_window_{window}"] = pp_direction_consistency(xyz, window)
        feats[f"ribose_plane_cons_window_{window}"] = ribose_plane_consistency(xyz, window)
        feats[f"base_stack_cons_window_{window}"] = base_stack_consistency(xyz, window)
        feats[f"bond_PO_cv_window_{window}"] = bond_PO3_cv(xyz, window)
        feats[f"radius_cv_window_{window}"] = backbone_radius_cv(xyz, window)
        feats[f"lambda2_std_window_{window}"] = lambda2_std(xyz, window)
        feats[f"tilt_sigma_window_{window}"] = tilt_sigma(xyz, window)
        feats[f"dm_frob_cons_window_{window}"] = dm_frobenius_consistency(xyz, window)

    return feats


def backbone_direction_vec(xyz, window=5):
    """
    xyz : ndarray, shape (L, 7, 3)
          每个残基 7 个原子坐标，index 0 = P，index 5 = O3'
    window : int
          滑窗大小（奇数更直观；偶数也可，会自动取对称区间）
    return : ndarray, shape (L,)
          每个残基局部骨架方向一致性 (0‑1)
    """
    L = xyz.shape[0]
    if L <= 2:                       # 边界情况
        return np.zeros(L, dtype=float)

    # 1) 先计算相邻 P → O3' 单位向量 (L‑1, 3)
    bond_vecs = xyz[1:, 5] - xyz[:-1, 0]
    bond_vecs /= np.linalg.norm(bond_vecs, axis=1, keepdims=True) + 1e-12

    # 2) 对每条向量中心 i，取窗口内 [i‑w2, i+w2] 所有向量两两余弦
    w2 = window // 2
    cons = np.zeros(L - 1)           # 结果先对应向量数 (L‑1,)
    for i in range(L - 1):
        left = max(0, i - w2)
        right = min(L - 2, i + w2)   # L‑2 是最后一条向量索引
        sub = bond_vecs[left:right + 1]  # (m,3)
        if len(sub) == 1:            # 只有自己
            cons[i] = 1.0
        else:
            # 逐对计算余弦并求平均
            total = 0.0
            cnt = 0
            for a in range(len(sub)):
                for b in range(a + 1, len(sub)):
                    total += float(np.dot(sub[a], sub[b]))
                    cnt += 1
            cons[i] = total / cnt

    # 3) 把 (L‑1,) 向量级一致性对齐到 L 个残基
    out = np.zeros(L, dtype=float)
    out[:-1] = cons        # 第 0..L‑2 残基
    out[-1] = cons[-1]     # 最后一个残基沿用前值，也可设 0
    out = np.clip(out, 0.0, 1.0)
    return out

def local_electrostatic_clustering_vec(xyz, window=11):
    """ 向量化滑动窗口静电势聚集 (L,) """
    P = xyz[:,0]  # (L,3)
    dist_matrix = cdist(P, P)  # (L,L)
    np.fill_diagonal(dist_matrix, np.inf)
    inv_dist = 1 / (dist_matrix + 1e-8)
    
    # 卷积实现滑动窗口平均
    kernel = np.ones(window)/window
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=1, arr=inv_dist
    ).diagonal()  # (L,)

def base_stacking_density_vec(xyz, window=5):
    """ 向量化滑动窗口碱基堆积评分 (L,) """
    if len(xyz) < 2:
        return np.zeros(len(xyz))
    c4s, bases = xyz[:,3], xyz[:,6]  # (L,3)
    normals = bases - c4s
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    
    # 计算相邻残基几何关系 (L-1,)
    dists = np.linalg.norm(bases[1:] - bases[:-1], axis=1)
    angles = np.arccos(np.clip(np.sum(normals[1:] * normals[:-1], axis=1), -1, 1))
    scores = np.exp(-0.5*((dists/3.4)**2 + (angles/0.5)**2))  # (L-1,)
    
    # 滑动窗口平均 (边界用最近值填充)
    scores_padded = np.pad(scores, (window//2, window//2), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(scores_padded, window)
    return np.concatenate([[0], windows.mean(axis=1), [0]])[:len(xyz)]  # (L,)

def dynamic_betweenness_vec(xyz, window=21, threshold=12.0):
    """ 向量化滑动窗口子图路径计数 (L,) """
    if len(xyz) < 2:
        return np.zeros(len(xyz))
    c4s = xyz[:,3]  # (L,3)
    adj = (cdist(c4s, c4s) < threshold).astype(float)  # (L,L)
    np.fill_diagonal(adj, 0)
    
    # 预计算所有可能的子图 (L,window,window)
    adj_padded = np.pad(adj, ((window//2,window//2),(window//2,window//2)), 'constant')
    windows = np.lib.stride_tricks.sliding_window_view(adj_padded, (window,window))  # (L,L,window,window)
    subgraphs = windows[np.arange(len(xyz)), np.arange(len(xyz))]  # (L,window,window)
    
    # 计算3跳路径总数 (近似介数)
    paths = np.linalg.matrix_power(subgraphs, 3)  # (L,window,window)
    return np.sum(paths, axis=(1,2)) / (window**2)  # (L,)

def local_helical_score_vec(xyz, window=5):
    """ 向量化计算滑动窗口内C4'的螺旋拟合误差 (L,) """
    c4s = xyz[:,3]  # (L,3)
    L = len(c4s)
    
    # 生成滑动窗口 (L,window,3)
    c4s_padded = np.pad(c4s, ((window//2,window//2),(0,0)), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(c4s_padded, (window,3))[:,0]
    
    # 计算每个窗口的协方差矩阵最小特征值 (L,3,3)
    centered = windows - np.mean(windows, axis=1, keepdims=True)
    cov = np.einsum('lwi,lwj->lij', centered, centered) / window
    
    # 特征值分解 (仅需最小特征值)
    eigvals = np.linalg.eigvalsh(cov)
    radii = np.sqrt(eigvals[:,0] + eigvals[:,1])  # (L,)
    return 1 / (radii + 1e-8)

def backbone_torsion_variance(xyz, window=5):
    """ 滑动窗口内P-C4'-C3'-O3'二面角的标准差 (L,) """
    P, C4, C3, O3 = xyz[:,0], xyz[:,3], xyz[:,4], xyz[:,5]  # (L,3)
    
    # 计算二面角（向量化实现）
    v1 = P - C4
    v2 = C3 - C4
    v3 = O3 - C3
    n1 = np.cross(v1, v2)  # (L,3)
    n2 = np.cross(v2, v3)  # (L,3)
    angles = np.arctan2(
        np.sum(np.cross(n1, n2) * v2/np.linalg.norm(v2,axis=1,keepdims=True), axis=1),
        np.sum(n1 * n2, axis=1)
    )  # (L,)
    
    # 滑动窗口标准差
    angles_padded = np.pad(angles, (window//2, window-1-window//2), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(angles_padded, window)
    return np.std(windows, axis=1)  # (L,)

def sugar_ring_distortion(xyz):
    """ 计算C4'-C3'-O3'平面的RMSD拟合误差 (L,) """
    c4, c3, o3 = xyz[:,3], xyz[:,4], xyz[:,5]  # (L,3) each
    plane_normals = np.cross(c3 - c4, o3 - c4)  # (L,3)
    plane_normals /= np.linalg.norm(plane_normals, axis=1, keepdims=True) + 1e-8
    
    # 计算所有骨架原子到平面的平均距离
    backbone_atoms = xyz[:,:6].reshape(-1,3)  # (L*6,3)
    expanded_normals = np.repeat(plane_normals, 6, axis=0)  # (L*6,3)
    distances = np.abs(np.sum((backbone_atoms - np.repeat(c4,6,axis=0)) * expanded_normals, axis=1))
    return distances.reshape(-1,6).mean(axis=1)  # (L,)

def base_orientation(xyz):
    """ 碱基平面法向量与糖环法向量的点积 (L,) """
    c4, base = xyz[:,3], xyz[:,6]  # (L,3)
    sugar_normals = np.cross(xyz[:,4] - c4, xyz[:,5] - c4)  # 糖环法向量
    base_normals = base - c4  # 碱基近似法向量
    
    # 归一化后计算点积
    sugar_normals /= np.linalg.norm(sugar_normals, axis=1, keepdims=True) + 1e-8
    base_normals /= np.linalg.norm(base_normals, axis=1, keepdims=True) + 1e-8
    return np.sum(sugar_normals * base_normals, axis=1)  # [-1,1]区间


def frenet_curvature_torsion(P):
    """
    P: (L,3)   磷原子坐标序列
    Returns:
      kappa:   shape (L,)      曲率 κ，首尾两个元素为 0
      torsion: shape (L,)      扭率 τ，首尾三个元素为 0
    """
    L = P.shape[0]
    kappa = np.zeros(L)
    torsion = np.zeros(L)

    if L < 4:
        return kappa, torsion

    # —— 曲率 κ：基于三点向量夹角 —— 
    # v1[i] = P[i+1] - P[i], v2[i] = P[i+2] - P[i+1], 共 (L-2) 段
    v1 = P[1:-1] - P[:-2]      # (L-2,3)
    v2 = P[2:]   - P[1:-1]     # (L-2,3)

    # 计算 cosθ = (v1·v2)/(|v1||v2|)，再 arccos 得到角度
    dot12 = np.einsum("ij,ij->i", v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot12 / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    angles = np.arccos(cos_theta)    # (L-2,)
    kappa[1:-1] = angles

    # —— 扭率 τ：基于四点二面角 —— 
    # 定义四段向量 b1,b2,b3
    b1 = P[1:-2] - P[:-3]    # (L-3,3)
    b2 = P[2:-1] - P[1:-2]   # (L-3,3)
    b3 = P[3:  ] - P[2:-1]   # (L-3,3)

    # 法向量 n1, n2
    n1 = np.cross(b1, b2)    # (L-3,3)
    n2 = np.cross(b2, b3)    # (L-3,3)

    # 归一化 b2 用于计算 m1
    b2_norm = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)  # (L-3,3)
    m1 = np.cross(n1, b2_norm)  # (L-3,3)

    # 计算 x = n1·n2, y = m1·n2
    x = np.einsum("ij,ij->i", n1, n2)   # (L-3,)
    y = np.einsum("ij,ij->i", m1, n2)   # (L-3,)

    # atan2(y, x) 得到二面角
    tors = np.arctan2(y, x)             # (L-3,)
    # 放置到 torsion 数组：对应 i from 1 to L-3 映射到 torsion[2:L-1]
    torsion[2:-1] = tors

    return kappa, torsion

def contact_number(center, cutoff=6.0):
    """
    简易邻域计数：统计每个残基重心 6Å 内其它残基数
    """
    dist = cdist(center, center)
    neigh = (dist < cutoff).astype(int) - np.eye(len(center), dtype=int)
    return neigh.sum(1)


def local_density(xyz, k=5):
    """ 向量化k近邻密度 (L,) """
    if len(xyz) <= 5:
        return np.zeros(len(xyz))
    c4s = xyz[:,3]  # (L,3)
    dists = cdist(c4s, c4s)  # (L,L)
    np.fill_diagonal(dists, np.inf)  # 忽略自身
    kth_dists = np.partition(dists, k, axis=1)[:, k]  # 第k近邻距离
    return 1 / (kth_dists + 1e-8)  # (L,)

def ss_tendency(xyz, window=3):
    """ 向量化二级结构倾向性 (L,) """
    c4s = xyz[:,3]  # (L,3)
    L = len(c4s)
    
    # 计算所有连续窗口的向量
    vecs = c4s[window:] - c4s[:-window]  # (L-window,3)
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    
    # 计算连续向量间的角度 (使用点积)
    cos_angles = np.sum(vecs_norm[1:] * vecs_norm[:-1], axis=1)  # (L-window-1,)
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    
    # 计算滑动窗口标准差
    if len(angles) >= window:
        # 使用as_strided实现滑动窗口
        shape = (len(angles) - window + 1, window)
        strides = (angles.strides[0], angles.strides[0])
        windows = np.lib.stride_tricks.as_strided(angles, shape=shape, strides=strides)
        stds = np.std(windows, axis=1)
    else:
        stds = np.array([])
    
    # 构建输出 (边缘填充0)
    features = np.zeros(L)
    start_idx = window + 1
    end_idx = start_idx + len(stds)
    features[start_idx:end_idx] = stds
    return features

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import svd


class RNAGeometry:
    """RNA几何特征计算工具集 (全静态方法)"""

    # ==================== 原子索引常量 ====================
    P = 0
    O5 = 1
    C5 = 2
    C4 = 3
    C3 = 4
    O3 = 5
    C2 = 6

    # ==================== 核心计算方法 ====================
    @staticmethod
    def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
        """计算四个点的二面角 (单位: 弧度)"""
        b0, b1, b2 = a - b, c - b, d - c
        n1 = np.cross(b0, b1)
        n2 = np.cross(b1, b2)
        y = np.dot(np.cross(n1, n2), b1) / (np.linalg.norm(b1) + 1e-10)
        x = np.dot(n1, n2)
        return np.arctan2(y, x)

    @staticmethod
    def compute_puckering(sugar_atoms: np.ndarray) -> Tuple[float, float]:
            
        """
        计算糖环折叠参数 (振幅, 相位角)
        输入: (5,3) 糖环原子坐标 [C1', C2', C3', C4', O4']
        """
        v = sugar_atoms - np.mean(sugar_atoms, axis=0)
        phases = 2 * np.pi * np.arange(5) / 5
        q = np.array([np.sum([v[j] * np.cos(m * phases[j]) for j in range(5)], axis=0) for m in range(3)])
        amplitude = np.linalg.norm(q[1])
        phase = np.arctan2(q[1, 1], q[1, 0])
        return amplitude, phase

    @staticmethod
    def base_orientation(base_atoms: np.ndarray, backbone_atoms: np.ndarray) -> np.ndarray:
        """
        计算碱基朝向向量
        输入:
            base_atoms: (N,3) 碱基原子坐标
            backbone_atoms: (M,3) 主链原子坐标
        返回: (3,) 单位向量
        """
        base_center = np.mean(base_atoms, axis=0)
        backbone_center = np.mean(backbone_atoms, axis=0)
        vec = base_center - backbone_center
        return vec / (np.linalg.norm(vec) + 1e-10)

    # ==================== 高级特征计算 ====================
    @staticmethod
    def helical_parameters(prev_res: np.ndarray, curr_res: np.ndarray, next_res: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算螺旋参数 (Rise, Twist, Roll, Tilt)
        输入:
            prev_res: (7,3) 前一个残基坐标
            curr_res: (7,3) 当前残基坐标
            next_res: (7,3) 后一个残基坐标 (可选)
        返回: 参数字典
        """
        # 当前碱基平面法向量
        curr_normal = np.cross(curr_res[RNAGeometry.C4] - curr_res[RNAGeometry.P], curr_res[RNAGeometry.C2] - curr_res[RNAGeometry.C4])
        curr_normal /= np.linalg.norm(curr_normal) + 1e-10

        # 前一个碱基法向量
        prev_normal = np.cross(prev_res[RNAGeometry.C4] - prev_res[RNAGeometry.P], prev_res[RNAGeometry.C2] - prev_res[RNAGeometry.C4])
        prev_normal /= np.linalg.norm(prev_normal) + 1e-10

        # Rise (沿螺旋轴位移)
        helix_axis = curr_res[RNAGeometry.C4] - prev_res[RNAGeometry.C4]
        helix_axis /= np.linalg.norm(helix_axis) + 1e-10
        rise = np.dot(curr_res[RNAGeometry.C4] - prev_res[RNAGeometry.C4], (prev_normal + curr_normal) / 2)

        # Twist (法向量间夹角)
        twist = np.arccos(np.clip(np.dot(prev_normal, curr_normal), -1, 1))

        # Roll/Tilt
        n_perp = curr_normal - np.dot(curr_normal, helix_axis) * helix_axis
        roll = np.arccos(np.clip(np.dot(prev_normal, n_perp) / (np.linalg.norm(prev_normal) * np.linalg.norm(n_perp) + 1e-10), -1, 1))

        tilt = np.arcsin(np.clip(np.dot(curr_normal - prev_normal, helix_axis), -1, 1))

        return {"rise": rise, "twist": np.degrees(twist), "roll": np.degrees(roll), "tilt": np.degrees(tilt)}

    @staticmethod
    def gaussian_curvature(patch: np.ndarray) -> float:
        """
        计算局部曲面的高斯曲率
        输入: (N,3) 点云坐标 (至少3个点)
        """
        centered = patch - np.mean(patch, axis=0)
        _, s, _ = svd(centered)
        return s[-1] / (s[0] + 1e-10)

    @staticmethod
    def hydration_sites(coord: np.ndarray, 
                       probe_radius: float=1.5) -> np.ndarray:
        """
        预测可能的水合位点 (基于分子表面可及性)
        返回: (L,) 每个残基的水合位点数量
        """
        L = coord.shape[0]
        hydration = np.zeros(L)
        
        # 计算所有P和O原子坐标
        polar_atoms = np.vstack([
            coord[:, 0, :],  # P
            coord[:, 1, :],  # O5'
            coord[:, 5, :]   # O3'
        ])
        
        # 计算极性原子间的距离矩阵
        dist_mat = squareform(pdist(polar_atoms))
        
        for i in range(L):
            # 当前残基的极性原子索引
            idx = [i, L+i, 2*L+i]  # P, O5', O3'
            
            # 计算可及表面积
            n_neighbors = np.sum(dist_mat[idx] < 2*probe_radius, axis=1)
            hydration[i] = np.sum(n_neighbors < 3)  # 暴露的极性原子数
        
        return hydration

    @staticmethod
    def base_stacking(coord: np.ndarray, window: int=3) -> np.ndarray:
        """
        计算碱基堆积强度 (基于平面距离和法向量夹角)
        返回: (L,) 数组
        """
        L = coord.shape[0]
        stacking = np.zeros(L)
        normals = []
        
        # 先计算所有碱基平面法向量
        for i in range(L):
            plane_atoms = coord[i, [3,6,0]]  # C4'-C2'-P 定义平面
            v1 = plane_atoms[1] - plane_atoms[0]
            v2 = plane_atoms[2] - plane_atoms[0]
            normals.append(np.cross(v1, v2))
        
        # 计算相邻碱基堆积
        for i in range(1, L):
            dist = np.linalg.norm(coord[i,6] - coord[i-1,6])  # C2'间距
            angle = np.arccos(np.clip(
                np.dot(normals[i], normals[i-1])/(
                np.linalg.norm(normals[i])*np.linalg.norm(normals[i-1]) + 1e-10), 
                -1, 1
            ))
            stacking[i] = np.exp(-dist/3.4) * np.cos(angle)  # 3.4Å是典型堆积距离
        
        return stacking

    # ==================== 特征集成方法 ====================
    @staticmethod
    def extract_all(coord: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
        """
        集成所有特征计算
        输入: (L,7,3) 原子坐标
        返回: 特征字典
        """
        L = coord.shape[0]
        features = {}

        # 糖环折叠角
        puckering = np.zeros((L, 2))
        for i in range(L):
            sugar_atoms = np.stack([coord[i, RNAGeometry.C2], coord[i, RNAGeometry.C4], coord[i, RNAGeometry.C3], coord[i, RNAGeometry.C5], coord[i, RNAGeometry.O5]])
            puckering[i] = RNAGeometry.compute_puckering(sugar_atoms)

        for i in range(puckering.shape[1]):
            features[f"puckering_{i}"] = puckering[:, i]

        # 螺旋参数
        helical = np.zeros((L, 4))
        for i in range(1, L):
            params = RNAGeometry.helical_parameters(coord[i - 1], coord[i])
            helical[i] = list(params.values())
            
        for i in range(helical.shape[1]):
            features[f"helical_{i}"] = helical[:, i]

        # 高斯曲率 (基于C4'原子)
        curvature = np.zeros(L)
        for i in range(window_size, L - window_size):
            patch = coord[i - window_size : i + window_size + 1, RNAGeometry.C4]
            curvature[i] = RNAGeometry.gaussian_curvature(patch)
        features["C4_curvature"] = curvature

        features.update({
            'stacking': RNAGeometry.base_stacking(coord),
            'hydration': RNAGeometry.hydration_sites(coord),
        })

        return features


def build_graph(coord):
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore")

    pos = torch.tensor(coord).mean(1)
    edge_index = knn_graph(pos, k=32)
    data_dict, edge_feats = extract_statistical_features(coord, edge_index.numpy())
    df = pd.DataFrame(index=range(len(coord)))

    for key, values in data_dict.items():
        try:
            df.loc[:, key] = values
        except:
            df.loc[:, key] = np.nan

    if len(edge_feats) > 0:
        edge_feats_df = pd.DataFrame(edge_feats, columns=[f"edge_feats_{i}" for i in range(edge_feats.shape[1])])
        df = pd.concat([df, edge_feats_df], axis=1)

    return df

def read_fasta_biopython(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def transform_coord_seq(seq, coord):
    ((name, seqs),) = seq.items()
    feat = build_graph(coord)

    feat["y"] = list(seqs)
    feat["pdb"] = name
    return feat

def generate_dataframe(fname):
    coord = np.nan_to_num(np.load(os.path.join(coords_dir, fname)))
    seq = read_fasta_biopython(os.path.join(seqs_dir, fname.replace("npy", "fasta")))
    now_feat = transform_coord_seq(seq, coord)
    return now_feat


if __name__ == '__main__':
    seqs_dir = "./RNA_design_public/RNAdesignv1/train/seqs/"
    coords_dir = "./RNA_design_public/RNAdesignv1/train/coords"

    coords = sorted(os.listdir(coords_dir))

    from joblib import Parallel, delayed

    feats = pd.concat(Parallel(backend="loky", n_jobs=128)(delayed(generate_dataframe)(fname) for i, fname in enumerate(tqdm(coords))), axis=0, ignore_index=True)
    print(feats.head())

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import GroupKFold

    df = feats
    models = []
    # 特征列（所有 'pos_' 开头的列）
    feature_cols = [col for col in df.columns if col not in ["pdb", "y"]]

    X = df[feature_cols].values  # 特征矩阵
    y = df["y"].values  # 目标变量（多分类标签）
    groups = df["pdb"].values  # 用于 GroupKFold 的分组列（例如 PDB ID）

    # 将标签转换为类别编码（XGBoost 需要从 0 开始的整数）
    ID2BASE = {0: "A", 1: "U", 2: "C", 3: "G"}
    BASE2ID = {"A": 0, "U": 1, "C": 2, "G": 3}
    y_encoded = df["y"].map(BASE2ID).values
    num_classes = len(BASE2ID)

    # GroupKFold 交叉验证
    n_splits = 5  # 5 折交叉验证
    group_kfold = GroupKFold(n_splits=n_splits)

    # XGBoost 参数（启用 GPU 和多分类）
    params = {
        "objective": "multi:softprob",  # 多分类输出概率
        "num_class": num_classes,
        "eval_metric": "mlogloss",  # 多分类对数损失
        "tree_method": "gpu_hist",  # 使用GPU加速
        "gpu_id": 0,
        "max_depth": 12,
        "learning_rate": 0.1,
        "min_child_weight": 5,  # 防止过拟合（类似 min_data_in_leaf）
        # "gamma": 0.1,  # 分裂最小增益
        # "lambda": 1.5,  # L2 正则（强于默认值 1）
        # "alpha": 0.1,  # L1 正则
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "n_jobs": 40,
    }

    # 存储每折的结果
    oof_preds = np.zeros((len(df), num_classes))
    test_preds = []
    scores = []

    # GroupKFold 训练和验证
    for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(X, y_encoded, groups)):
        print(f"\nFold {fold + 1}")
        X_train, y_train = X[train_idx], y_encoded[train_idx]
        X_valid, y_valid = X[valid_idx], y_encoded[valid_idx]

        # XGBoost 数据集
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        # 验证集预测
        oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
        valid_score = log_loss(y_valid, oof_preds[valid_idx])
        valid_acc = accuracy_score(y_valid, np.argmax(oof_preds[valid_idx], axis=1))
        scores.append((valid_score, valid_acc))
        print(f"Valid Log Loss: {valid_score:.4f}, Valid Accuracy: {valid_acc:.4f}")
        models.append(model)

    # 输出交叉验证结果
    logloss_scores, acc_scores = zip(*scores)
    print(f"\nMean Log Loss: {np.mean(logloss_scores):.4f}")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f}")

    look = df[["y", "pdb"]]
    look["pred"] = np.argmax(oof_preds, axis=1)
    look["pred"] = look["pred"].map(ID2BASE)
    recover_rate = look.groupby("pdb").apply(lambda x: (x["y"] == x["pred"]).mean()).mean()
    print("Mean Recover Rate:", recover_rate)

    import joblib

    for i, m in enumerate(models):
        joblib.dump(m, f"./xgb_model_v1/xgb_model_numpy_balanced_{n_splits}fold_{i}_{round(recover_rate, 4)}.pkl")
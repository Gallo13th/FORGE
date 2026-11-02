import os
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import sparse
# ----------------------------
# 特征名
# ----------------------------

NODEFEAT_NAMES = ['bond_P-O5', 'bond_O5-C5', 'bond_C5-C4', 'bond_C4-C3', 'bond_C3-O3', 'bond_O3-N', 'dist_P-C5', 'dist_P-C4', 'dist_O5-C4', 'dist_O5-C3', 'dist_C5-C3', 'dist_C5-O3', 'dist_C4-O3', 'dist_C4-N', 'dist_C3-N', 'dist_P-P_shift1', 'dist_P-P_shift2', 'dist_P-P_shift3', 'dist_P-O5_shift1', 'dist_P-O5_shift2', 'dist_P-O5_shift3', 'dist_C4-C4_shift1', 'dist_C4-C4_shift2', 'dist_C4-C4_shift3', 'dist_C4-P_shift1', 'dist_C4-P_shift2', 'dist_C4-P_shift3', 'angle_P_P_P_span1', 'angle_P_P_P_span2', 'angle_P_P_P_span3', 'angle_P_C4_P_span1', 'angle_P_C4_P_span2', 'angle_P_C4_P_span3', 'angle_C4_C4_C4_span1', 'angle_C4_C4_C4_span2', 'angle_C4_C4_C4_span3', 'sugar_planarity', 'delta_P_x', 'delta_P_y', 'delta_P_z', 'delta2_P_x', 'delta2_P_y', 'delta2_P_z', 'delta_O5_x', 'delta_O5_y', 'delta_O5_z', 'delta2_O5_x', 'delta2_O5_y', 'delta2_O5_z', 'delta_C5_x', 'delta_C5_y', 'delta_C5_z', 'delta2_C5_x', 'delta2_C5_y', 'delta2_C5_z', 'delta_C4_x', 'delta_C4_y', 'delta_C4_z', 'delta2_C4_x', 'delta2_C4_y', 'delta2_C4_z', 'delta_C3_x', 'delta_C3_y', 'delta_C3_z', 'delta2_C3_x', 'delta2_C3_y', 'delta2_C3_z', 'delta_O3_x', 'delta_O3_y', 'delta_O3_z', 'delta2_O3_x', 'delta2_O3_y', 'delta2_O3_z', 'delta_N_x', 'delta_N_y', 'delta_N_z', 'delta2_N_x', 'delta2_N_y', 'delta2_N_z', 'center_x', 'center_y', 'center_z', 'p_dist_mean', 'p_dist_std', 'p_dist_skew', 'p_dist_kurtosis', 'P_kappa_curvature', 'O5_kappa_curvature', 'C5_kappa_curvature', 'C4_kappa_curvature', 'C3_kappa_curvature', 'O3_kappa_curvature', 'N_kappa_curvature', 'dihedral_sin', 'dihedral_cos', 'p_window_std', 'sasa_4A', 'vec_p_center_x', 'vec_p_center_y', 'vec_p_center_z', 'inertia_min', 'inertia_max', 'inertia_ratio', 'dist_global_ctr', 'dist_main_dim', 'puckering_0', 'puckering_1', 'helical_rise', 'helical_twist', 'helical_roll', 'helical_tilt', 'C4_local_curvature_5', 'sequential_turn_span_angle', 'stacking', 'hydration', 'backbone_dir_cons5', 'backbone_dir_cons11', 'elec_cluster6', 'elec_cluster15', 'stack_density5', 'stack_density11', 'dyn_betw10', 'dyn_betw21', 'local_helix5', 'local_helix11', 'torsion_var5', 'torsion_var7', 'ring_distort', 'base_orient', 'P_kappa', 'P_tau', 'C4_kappa', 'C4_tau', 'O5_kappa', 'O5_tau', 'local_density', 'contact_num', 'ss_tendency', 'tri_P_O3prev_O5', 'tri_P_O5_C5', 'tri_C4_C5_C3', 'tri_C4_C3_O3', 'tri_tau_P_C4_next', 'va_P_O5_vs_C5_C4', 'va_base_vs_backbone', 'pp_dir_cons_3', 'ribose_plane_cons_3', 'base_stack_cons_3', 'bond_PO_cv_3', 'radius_cv_3', 'lambda2_std_3', 'tilt_sigma_3', 'dm_frob_cons_3', 'tor_smooth_3', 'pp_dir_cons_5', 'ribose_plane_cons_5', 'base_stack_cons_5', 'bond_PO_cv_5', 'radius_cv_5', 'lambda2_std_5', 'tilt_sigma_5', 'dm_frob_cons_5', 'tor_smooth_5', 'pp_dir_cons_7', 'ribose_plane_cons_7', 'base_stack_cons_7', 'bond_PO_cv_7', 'radius_cv_7', 'lambda2_std_7', 'tilt_sigma_7', 'dm_frob_cons_7', 'tor_smooth_7', 'pp_dir_cons_11', 'ribose_plane_cons_11', 'base_stack_cons_11', 'bond_PO_cv_11', 'radius_cv_11', 'lambda2_std_11', 'tilt_sigma_11', 'dm_frob_cons_11', 'tor_smooth_11', 'pp_dir_cons_13', 'ribose_plane_cons_13', 'base_stack_cons_13', 'bond_PO_cv_13', 'radius_cv_13', 'lambda2_std_13', 'tilt_sigma_13', 'dm_frob_cons_13', 'tor_smooth_13']

EDGEFEAT_NAMES = ['dist_P-P_rbf0', 'dist_P-P_rbf1', 'dist_P-P_rbf2', 'dist_P-P_rbf3', 'dist_P-P_rbf4', 'dist_P-P_rbf5', 'dist_P-P_rbf6', 'dist_P-P_rbf7', 'dist_P-P_rbf8', 'dist_P-P_rbf9', 'dist_P-P_rbf10', 'dist_P-P_rbf11', 'dist_P-P_rbf12', 'dist_P-P_rbf13', 'dist_P-P_rbf14', 'dist_P-P_rbf15', 'dist_P-P_rbf16', 'dist_P-P_rbf17', 'dist_P-P_rbf18', 'dist_P-P_rbf19', 'dist_P-P_rbf20', 'dist_P-P_rbf21', 'dist_P-P_rbf22', 'dist_P-P_rbf23', 'dist_P-P_rbf24', 'dist_P-P_rbf25', 'dist_P-P_rbf26', 'dist_P-P_rbf27', 'dist_P-P_rbf28', 'dist_P-P_rbf29', 'dist_P-P_rbf30', 'dist_P-P_rbf31', 'dist_O5-P_rbf0', 'dist_O5-P_rbf1', 'dist_O5-P_rbf2', 'dist_O5-P_rbf3', 'dist_O5-P_rbf4', 'dist_O5-P_rbf5', 'dist_O5-P_rbf6', 'dist_O5-P_rbf7', 'dist_O5-P_rbf8', 'dist_O5-P_rbf9', 'dist_O5-P_rbf10', 'dist_O5-P_rbf11', 'dist_O5-P_rbf12', 'dist_O5-P_rbf13', 'dist_O5-P_rbf14', 'dist_O5-P_rbf15', 'dist_O5-P_rbf16', 'dist_O5-P_rbf17', 'dist_O5-P_rbf18', 'dist_O5-P_rbf19', 'dist_O5-P_rbf20', 'dist_O5-P_rbf21', 'dist_O5-P_rbf22', 'dist_O5-P_rbf23', 'dist_O5-P_rbf24', 'dist_O5-P_rbf25', 'dist_O5-P_rbf26', 'dist_O5-P_rbf27', 'dist_O5-P_rbf28', 'dist_O5-P_rbf29', 'dist_O5-P_rbf30', 'dist_O5-P_rbf31', 'dist_C5-P_rbf0', 'dist_C5-P_rbf1', 'dist_C5-P_rbf2', 'dist_C5-P_rbf3', 'dist_C5-P_rbf4', 'dist_C5-P_rbf5', 'dist_C5-P_rbf6', 'dist_C5-P_rbf7', 'dist_C5-P_rbf8', 'dist_C5-P_rbf9', 'dist_C5-P_rbf10', 'dist_C5-P_rbf11', 'dist_C5-P_rbf12', 'dist_C5-P_rbf13', 'dist_C5-P_rbf14', 'dist_C5-P_rbf15', 'dist_C5-P_rbf16', 'dist_C5-P_rbf17', 'dist_C5-P_rbf18', 'dist_C5-P_rbf19', 'dist_C5-P_rbf20', 'dist_C5-P_rbf21', 'dist_C5-P_rbf22', 'dist_C5-P_rbf23', 'dist_C5-P_rbf24', 'dist_C5-P_rbf25', 'dist_C5-P_rbf26', 'dist_C5-P_rbf27', 'dist_C5-P_rbf28', 'dist_C5-P_rbf29', 'dist_C5-P_rbf30', 'dist_C5-P_rbf31', 'dist_C4-P_rbf0', 'dist_C4-P_rbf1', 'dist_C4-P_rbf2', 'dist_C4-P_rbf3', 'dist_C4-P_rbf4', 'dist_C4-P_rbf5', 'dist_C4-P_rbf6', 'dist_C4-P_rbf7', 'dist_C4-P_rbf8', 'dist_C4-P_rbf9', 'dist_C4-P_rbf10', 'dist_C4-P_rbf11', 'dist_C4-P_rbf12', 'dist_C4-P_rbf13', 'dist_C4-P_rbf14', 'dist_C4-P_rbf15', 'dist_C4-P_rbf16', 'dist_C4-P_rbf17', 'dist_C4-P_rbf18', 'dist_C4-P_rbf19', 'dist_C4-P_rbf20', 'dist_C4-P_rbf21', 'dist_C4-P_rbf22', 'dist_C4-P_rbf23', 'dist_C4-P_rbf24', 'dist_C4-P_rbf25', 'dist_C4-P_rbf26', 'dist_C4-P_rbf27', 'dist_C4-P_rbf28', 'dist_C4-P_rbf29', 'dist_C4-P_rbf30', 'dist_C4-P_rbf31', 'dist_C3-P_rbf0', 'dist_C3-P_rbf1', 'dist_C3-P_rbf2', 'dist_C3-P_rbf3', 'dist_C3-P_rbf4', 'dist_C3-P_rbf5', 'dist_C3-P_rbf6', 'dist_C3-P_rbf7', 'dist_C3-P_rbf8', 'dist_C3-P_rbf9', 'dist_C3-P_rbf10', 'dist_C3-P_rbf11', 'dist_C3-P_rbf12', 'dist_C3-P_rbf13', 'dist_C3-P_rbf14', 'dist_C3-P_rbf15', 'dist_C3-P_rbf16', 'dist_C3-P_rbf17', 'dist_C3-P_rbf18', 'dist_C3-P_rbf19', 'dist_C3-P_rbf20', 'dist_C3-P_rbf21', 'dist_C3-P_rbf22', 'dist_C3-P_rbf23', 'dist_C3-P_rbf24', 'dist_C3-P_rbf25', 'dist_C3-P_rbf26', 'dist_C3-P_rbf27', 'dist_C3-P_rbf28', 'dist_C3-P_rbf29', 'dist_C3-P_rbf30', 'dist_C3-P_rbf31', 'dist_O3-P_rbf0', 'dist_O3-P_rbf1', 'dist_O3-P_rbf2', 'dist_O3-P_rbf3', 'dist_O3-P_rbf4', 'dist_O3-P_rbf5', 'dist_O3-P_rbf6', 'dist_O3-P_rbf7', 'dist_O3-P_rbf8', 'dist_O3-P_rbf9', 'dist_O3-P_rbf10', 'dist_O3-P_rbf11', 'dist_O3-P_rbf12', 'dist_O3-P_rbf13', 'dist_O3-P_rbf14', 'dist_O3-P_rbf15', 'dist_O3-P_rbf16', 'dist_O3-P_rbf17', 'dist_O3-P_rbf18', 'dist_O3-P_rbf19', 'dist_O3-P_rbf20', 'dist_O3-P_rbf21', 'dist_O3-P_rbf22', 'dist_O3-P_rbf23', 'dist_O3-P_rbf24', 'dist_O3-P_rbf25', 'dist_O3-P_rbf26', 'dist_O3-P_rbf27', 'dist_O3-P_rbf28', 'dist_O3-P_rbf29', 'dist_O3-P_rbf30', 'dist_O3-P_rbf31', 'dist_P-N_rbf0', 'dist_P-N_rbf1', 'dist_P-N_rbf2', 'dist_P-N_rbf3', 'dist_P-N_rbf4', 'dist_P-N_rbf5', 'dist_P-N_rbf6', 'dist_P-N_rbf7', 'dist_P-N_rbf8', 'dist_P-N_rbf9', 'dist_P-N_rbf10', 'dist_P-N_rbf11', 'dist_P-N_rbf12', 'dist_P-N_rbf13', 'dist_P-N_rbf14', 'dist_P-N_rbf15', 'dist_P-N_rbf16', 'dist_P-N_rbf17', 'dist_P-N_rbf18', 'dist_P-N_rbf19', 'dist_P-N_rbf20', 'dist_P-N_rbf21', 'dist_P-N_rbf22', 'dist_P-N_rbf23', 'dist_P-N_rbf24', 'dist_P-N_rbf25', 'dist_P-N_rbf26', 'dist_P-N_rbf27', 'dist_P-N_rbf28', 'dist_P-N_rbf29', 'dist_P-N_rbf30', 'dist_P-N_rbf31', 'direct_x', 'direct_y', 'direct_z', 'quat_0', 'quat_1', 'quat_2', 'quat_3', 'pe_0', 'pe_1', 'pe_2', 'pe_3', 'pe_4', 'pe_5', 'pe_6', 'pe_7', 'pe_8', 'pe_9', 'pe_10', 'pe_11', 'pe_12', 'pe_13', 'pe_14', 'pe_15', 'pe_16', 'pe_17', 'pe_18', 'pe_19', 'pe_20', 'pe_21', 'pe_22', 'pe_23', 'pe_24', 'pe_25', 'pe_26', 'pe_27', 'pe_28', 'pe_29', 'pe_30', 'pe_31', 'rosetta_0', 'rosetta_1', 'rosetta_2', 'rosetta_3', 'rosetta_4', 'rosetta_5', 'rosetta_6', 'rosetta_7', 'rosetta_8', 'rosetta_9', 'rosetta_10', 'rosetta_11', 'rosetta_12', 'rosetta_13', 'rosetta_14', 'rosetta_15', 'rosetta_16', 'rosetta_17', 'rosetta_18', 'rosetta_19', 'rosetta_20', 'rosetta_21', 'rosetta_22', 'rosetta_23', 'rosetta_24', 'rosetta_25', 'rosetta_26', 'rosetta_27', 'rosetta_28', 'rosetta_29', 'rosetta_30', 'rosetta_31', 'rosetta_32', 'rosetta_33', 'rosetta_34', 'rosetta_35', 'rosetta_36', 'rosetta_37', 'rosetta_38', 'rosetta_39', 'cec_0', 'cec_1', 'cec_2', 'cec_3', 'cec_4', 'cec_5', 'cec_6', 'cec_7', 'pair_angle_1', 'pair_angle_2', 'pair_angle_3', 'pair_angle_4', 'dih_0_0', 'dih_0_1', 'dih_1_0', 'dih_1_1', 'dih_2_0', 'dih_2_1', 'dih_3_0', 'dih_3_1', 'dih_4_0', 'dih_4_1', 'dih_5_0', 'dih_5_1', 'dih_6_0', 'dih_6_1', 'tri_0_0', 'tri_0_1', 'tri_1_0', 'tri_1_1', 'tri_2_0', 'tri_2_1', 'tri_3_0', 'tri_3_1', 'tri_4_0', 'tri_4_1', 'tri_5_0', 'tri_5_1', 'tri_6_0', 'tri_6_1', 'stack_flag']


FEATNAMES = NODEFEAT_NAMES + [f'edge_{name}_mean' for name in EDGEFEAT_NAMES] + [f'edge_{name}_std' for name in EDGEFEAT_NAMES]

# ----------------------------
# 工具函数定义
# ----------------------------
def summarize_gain_importance(results_dir, clusters, feature_names, top_n=50):
    """
    汇总所有gain重要性分析结果并绘制热图
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 收集所有重要性数据
    all_importance = {}
    for cluster in clusters:
        # 同家族验证集重要性
        val_gain_file = results_dir + f"/{cluster}_val_gain_gain_importance.csv"
        # print(val_gain_file, os.path.exists(val_gain_file))
        if os.path.exists(val_gain_file):
            df = pd.read_csv(val_gain_file)
            all_importance[f"{cluster}_val"] = dict(zip(df['feature'], df['importance']))
        
        # 交叉测试重要性
        for test_cluster in clusters:
            if cluster != test_cluster:
                cross_gain_file = os.path.join(results_dir, f"{cluster}_on_{test_cluster}_gain_gain_importance.csv")
                if os.path.exists(cross_gain_file):
                    df = pd.read_csv(cross_gain_file)
                    all_importance[f"{cluster}_on_{test_cluster}"] = dict(zip(df['feature'], df['importance']))
    
    # 检查是否有数据
    if not all_importance:
        print("⚠️ 未找到任何gain重要性文件，跳过热图生成")
        return None
    
    # 选择top N特征（按总体重要性）
    total_importance = {}
    for importance_dict in all_importance.values():
        for feat, imp in importance_dict.items():
            total_importance[feat] = total_importance.get(feat, 0) + imp
    
    top_features = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    selected_features = [feat for feat, _ in top_features]
    
    # 构建重要性矩阵
    importance_matrix = []
    row_labels = []
    
    for key, importance_dict in all_importance.items():
        row_data = [importance_dict.get(feat, 0) for feat in selected_features]
        importance_matrix.append(row_data)
        row_labels.append(key)
    
    # 检查矩阵是否为空
    if not importance_matrix or not row_labels:
        print("⚠️ 重要性矩阵为空，跳过热图生成")
        return None
    
    # 转换为DataFrame
    importance_df = pd.DataFrame(
        importance_matrix, 
        index=row_labels, 
        columns=selected_features
    )
    
    # 绘制热图
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        importance_df, 
        cmap='YlOrRd', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True,
        vmin=0
    )
    plt.title('Feature Importance (Gain) Across All Tests', fontsize=16)
    plt.xlabel('Features (Top 50 by Overall Importance)', fontsize=12)
    plt.ylabel('Test Cases', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # 保存图片
    heatmap_path = os.path.join(results_dir, "gain_importance_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gain重要性热图已保存至 {heatmap_path}")
    
    # 保存重要性矩阵
    matrix_path = os.path.join(results_dir, "gain_importance_matrix.csv")
    importance_df.to_csv(matrix_path)
    print(f"📊 Gain重要性矩阵已保存至 {matrix_path}")
    
    return importance_df

import shap

def explain_model_with_shap(model, X_train, X_test, feature_names, save_path=None):
    """
    使用SHAP解释模型预测
    """
    print("🧮 正在使用SHAP解释模型...")
    
    # 使用训练集样本作为背景数据（建议使用子集以提高速度）
    background = shap.sample(X_train, 100) if X_train.shape[0] > 100 else X_train
    explainer = shap.Explainer(model.predict, background)
    
    # 解释测试集
    shap_values = explainer(X_test[:1000])  # 可选：只解释前1000个样本以节省时间
    
    # 保存SHAP值
    if save_path:
        np.savez_compressed(save_path, shap_values=shap_values.values, expected_value=explainer.expected_value)
        print(f"💾 SHAP值已保存至 {save_path}.npz")

    # 生成特征重要性
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
        # 多分类：取平均绝对SHAP值
        shap_importance = np.mean(np.abs(shap_values.values), axis=(0, 1))
    else:
        shap_importance = np.mean(np.abs(shap_values.values), axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    }).sort_values(by='importance', ascending=False)

    if save_path:
        importance_df.to_csv(f"{save_path}_importance.csv", index=False)
        print(f"📊 SHAP特征重要性已保存至 {save_path}_importance.csv")

    return shap_values, importance_df

def analyze_feature_importance_gain(model, feature_names, save_path=None):
    """
    使用XGBoost内置的gain分析特征重要性
    """
    print("📊 正在使用XGBoost gain分析特征重要性...")
    
    # 获取特征重要性（按gain）
    importance_scores = model.feature_importances_
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values(by='importance', ascending=False)
    
    # 保存结果
    if save_path:
        importance_df.to_csv(f"{save_path}_gain_importance.csv", index=False)
        print(f"📊 Gain特征重要性已保存至 {save_path}_gain_importance.csv")
    
    return importance_df

def aggregate_features_mean_vectorized(features, indices, num_nodes):
    if features.shape[0] == 0:
        return np.zeros((num_nodes, features.shape[1]))
    rows = indices
    cols = np.arange(len(indices))
    data = np.ones(len(indices))
    incidence_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, len(indices))).tocsr()
    degree = np.array(incidence_matrix.sum(axis=1)).flatten()
    sum_features = incidence_matrix @ features
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_features = sum_features / degree[:, np.newaxis]
    mean_features[degree == 0, :] = 0
    return mean_features


def aggregate_features_std_vectorized(features, indices, num_nodes):
    if features.shape[0] == 0:
        return np.zeros((num_nodes, features.shape[1]))
    mean_features = aggregate_features_mean_vectorized(features, indices, num_nodes)
    rows = indices
    node_means = mean_features[rows]
    diff = features - node_means
    squared_diff = diff ** 2
    cols = np.arange(len(indices))
    data = np.ones(len(indices))
    incidence_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, len(indices))).tocsr()
    sum_squared_diff = incidence_matrix @ squared_diff
    degree = np.array(incidence_matrix.sum(axis=1)).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        variance = sum_squared_diff / degree[:, np.newaxis]
    variance[degree == 0, :] = 0
    std_features = np.sqrt(variance)
    std_features[degree == 0, :] = 0
    return std_features


# ----------------------------
# 数据集定义
# ----------------------------

class QuantizedGraphDataset(Dataset):
    def __init__(self, df, root_dir="rfam_quantization/quantized/", lmdb_path='./lmdb_data', use_lmdb=True):
        self.df = df
        self.root_dir = root_dir
        self.seq_root_dir = "./seq/"
        self.lmdb_path = lmdb_path
        self.use_lmdb = use_lmdb

        if self.use_lmdb:
            self.env = self.check_or_create_lmdb()

    def __len__(self):
        return len(self.df)

    def check_or_create_lmdb(self):
        if not os.path.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path)
            print(f"LMDB directory {self.lmdb_path} created.")
            self.process_and_store_to_lmdb()
        else:
            print(f"LMDB directory {self.lmdb_path} already exists.")
            self._print_lmdb_size()

        env = lmdb.open(self.lmdb_path, map_size=20*1024*1024*1024, readonly=True, lock=False)
        return env

    def _print_lmdb_size(self):
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(self.lmdb_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024**2:
                size_str = f"{total_size/1024:.2f} KB"
            elif total_size < 1024**3:
                size_str = f"{total_size/1024**2:.2f} MB"
            else:
                size_str = f"{total_size/1024**3:.2f} GB"
            print(f"LMDB actual size: {size_str} ({total_size} bytes)")
        except Exception as e:
            print(f"Could not get LMDB size: {e}")

    def _dequantize_features(self, data, quant_params, log_flags):
        dequantized = data.copy()
        qmax = 2. ** 8 - 1.
        for feat_idx in quant_params.keys():
            min_val = quant_params[feat_idx]['min']
            max_val = quant_params[feat_idx]['max']
            scale = (max_val - min_val) / qmax
            dequantized[:, feat_idx] = dequantized[:, feat_idx] * scale + min_val
            if log_flags[feat_idx]:
                dequantized[:, feat_idx] = np.sign(dequantized[:, feat_idx]) * (np.exp(np.abs(dequantized[:, feat_idx])) - 1)
        return dequantized

    def _load_and_process_item(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['pdb_id'].upper()
        chain : str = row['chain']
        if chain.isupper():
            chain = 'H_' + chain
        elif chain.islower():
            chain = 'L_' + chain
        else:
            chain = 'X_' + chain
        pdb_head = f"{pdb_id}_{chain}"
        file_path = os.path.join(self.root_dir, f"{pdb_head}_quant.npz")

        if not os.path.exists(file_path):
            print(f"Quantized file {file_path} does not exist.")
            return None
        seq_file = os.path.join(self.seq_root_dir, f"{pdb_head}.fasta")
        try:
            with open(seq_file, 'r') as f:
                seq = f.readlines()[-1].strip()
            base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            label = [base_map.get(s, -1) for s in seq]
            if len(label) >= 500:
                print(f"Skipping {pdb_head} due to sequence length {len(label)} >= 1000")
                return None
        except Exception as e:
            print(f"Error reading {seq_file}: {e}")
            return None
        try:
            quant_data = np.load(file_path, allow_pickle=True)

            x = self._dequantize_features(
                quant_data['x'], 
                quant_data['x_quant_params'].item(), 
                quant_data['x_log_flag']
            )
        
            edge_attr = self._dequantize_features(
                quant_data['edge_attr'], 
                quant_data['edge_attr_quant_params'].item(), 
                quant_data['edge_attr_log_flag']
            )


            num_nodes = x.shape[0]
            edge_index = quant_data['edge_index']

            if edge_attr.shape[0] > 0 and edge_index.shape[1] > 0:
                edge_mean = aggregate_features_mean_vectorized(edge_attr, edge_index[1], num_nodes)
                edge_std = aggregate_features_std_vectorized(edge_attr, edge_index[1], num_nodes)
                edge_agg_features = np.concatenate([edge_mean, edge_std], axis=-1)
            else:
                edge_feat_dim = edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0
                edge_agg_features = np.zeros((num_nodes, edge_feat_dim * 2))

            all_features = np.nan_to_num(np.concatenate([x, edge_agg_features], axis=-1))
        
            return {
                'features': all_features,
                'labels': label,
                'pdb_head': pdb_head
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_and_store_to_lmdb(self):
        print("Starting to process and store data to LMDB...")
        lmdb_env = lmdb.open(self.lmdb_path, map_size=20*1024*1024*1024)
        
        success_count = 0
        with lmdb_env.begin(write=True) as txn:
            for idx in range(len(self.df)):
                item = self._load_and_process_item(idx)
                # print(item)
                if item is not None:
                    key = item['pdb_head'].encode()
                    value = pickle.dumps(item)
                    txn.put(key, value)
                    success_count += 1
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(self.df)} samples, {success_count} successful")

        lmdb_env.close()
        print(f"LMDB creation completed! Successfully processed {success_count}/{len(self.df)} samples")
        self._print_lmdb_size()

    def __getitem__(self, idx):
        if self.use_lmdb:
            chain: str = self.df.iloc[idx]['chain']
            if chain.isupper():
                chain = 'H_' + chain
            elif chain.islower():
                chain = 'L_' + chain
            else:
                chain = 'X_' + chain
            pdb_head = self.df.iloc[idx]['pdb_id'].upper() + '_' + chain
            with self.env.begin() as txn:
                raw_data = txn.get(pdb_head.encode())
                if raw_data is None:
                    return None
                item = pickle.loads(raw_data)
                inner_label = np.array(item['labels']) != -1
                item['labels'] = np.array(item['labels'])[inner_label]
                item['features'] = item['features'][inner_label]
                return item
        else:
            return self._load_and_process_item(idx)

    def prepare_lmdb(self):
        if os.path.exists(self.lmdb_path):
            print(f"LMDB already exists at {self.lmdb_path}")
            self._print_lmdb_size()
            return
        self.process_and_store_to_lmdb()


# ----------------------------
# 数据收集函数
# ----------------------------

def collect_data_from_dataloader(dataloader, desc="Processing", expected_feature_dim=None):
    X_list = []
    y_list = []
    valid_samples = 0
    invalid_samples = 0
    
    print(f"📌 正在{desc}数据...")
    for batch in tqdm(dataloader, desc=desc):
        for sample in batch:
            if sample is None:
                invalid_samples += 1
                continue
                
            features = sample['features']
            labels = sample['labels']
            
            # 检查特征维度是否一致
            if expected_feature_dim is None:
                expected_feature_dim = features.shape[1]
            
            if features.shape[1] != expected_feature_dim:
                print(f"⚠️  跳过样本 {sample.get('pdb_head', 'unknown')} - 特征维度不匹配: {features.shape[1]} vs {expected_feature_dim}")
                invalid_samples += 1
                continue
                
            # 检查标签是否有效
            if np.any(labels == -1):
                print(f"⚠️  跳过样本 {sample.get('pdb_head', 'unknown')} - 包含无效标签")
                invalid_samples += 1
                continue
                
            X_list.append(features)
            y_list.append(labels)
            valid_samples += 1
    
    print(f"📊 有效样本数: {valid_samples}, 无效样本数: {invalid_samples}")
    
    if len(X_list) > 0:
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        print(f"✅ 数据收集完成: X.shape = {X.shape}")
        return X, y
    else:
        print("❌ 数据为空！")
        return None, None


# ----------------------------
# 交叉测试函数
# ----------------------------

def cross_test_model(model, train_cluster, test_cluster, df, lmdb_path, results_dir):
    """用训练好的模型在其他cluster上进行测试"""
    print(f"\n🔄 交叉测试: 用 {train_cluster} 训练的模型测试 {test_cluster}")
    
    # 获取测试集索引
    test_indices = df[df['CL'] == test_cluster].index.tolist()
    if not test_indices:
        print(f"⚠️ 测试集 {test_cluster} 无样本，跳过测试。")
        return None
    
    # 创建测试数据集
    dataset = QuantizedGraphDataset(df, lmdb_path=lmdb_path, use_lmdb=True)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: x)
    
    # 收集测试数据（使用训练时的特征维度）
    expected_dim = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    X_test, y_test = collect_data_from_dataloader(test_loader, f"收集{test_cluster}测试数据", expected_dim)
    
    if X_test is None or y_test is None:
        print(f"⚠️ {test_cluster} 测试数据为空，跳过测试。")
        return None
    
    # 进行预测
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    # 在交叉测试数据上运行SHAP分析
    shap_save_path = os.path.join(results_dir, f"{train_cluster}_on_{test_cluster}_shap")
    # 替换原来的SHAP分析
    gain_save_path = os.path.join(results_dir, f"{train_cluster}_on_{test_cluster}_gain")
    analyze_feature_importance_gain(
        model=model,
        feature_names=FEATNAMES,
        save_path=gain_save_path
    )
    
    # 生成详细报告
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"✅ {test_cluster} 测试准确率: {test_acc:.4f}")
    
    # 保存交叉测试结果
    cross_test_result = {
        'train_cluster': train_cluster,
        'test_cluster': test_cluster,
        'accuracy': test_acc,
        'classification_report': report,
        'test_size': len(y_test)
    }
    
    # 保存结果
    result_file = os.path.join(results_dir, f"{train_cluster}_on_{test_cluster}_cross_test.json")
    with open(result_file, 'w') as f:
        json.dump(cross_test_result, f, indent=2)
    
    print(f"✅ 交叉测试结果已保存为 '{result_file}'")
    
    return cross_test_result


# ----------------------------
# 创建"All"类别数据集
# ----------------------------

def create_all_cluster_dataset(df, sampling_ratio=0.5):
    """从每个cluster中采样数据创建"All"类别数据集"""
    print(f"\n🎯 创建 'All' 类别数据集 (采样比例: {sampling_ratio})")
    
    all_indices = []
    
    # 获取所有cluster列表
    clusters = df['CL'].unique().tolist()
    print(f"📊 总共 {len(clusters)} 个clusters: {clusters}")
    
    for cluster in clusters:
        cluster_indices = df[df['CL'] == cluster].index.tolist()
        if cluster_indices:
            # 从每个cluster中采样指定比例的数据
            sample_size = max(1, int(len(cluster_indices) * sampling_ratio))
            sampled_indices = np.random.choice(cluster_indices, size=sample_size, replace=False).tolist()
            all_indices.extend(sampled_indices)
            print(f"  📦 {cluster}: {len(cluster_indices)} → {sample_size} 样本")
        else:
            print(f"  ⚠️  {cluster}: 无样本")
    
    print(f"📈 'All' 类别总计: {len(all_indices)} 样本")
    return all_indices


# ----------------------------
# 主训练函数（增加交叉测试）
# ----------------------------

def train_model_on_subset(df, subset_name, lmdb_path, save_dir, all_clusters, indices=None):
    print(f"\n🚀 开始训练子集: {subset_name}")

    # 如果没有提供indices，则按cluster名称获取
    if indices is None:
        indices = df[df['CL'] == subset_name].index.tolist()
    
    if not indices:
        print(f"⚠️ 子集 {subset_name} 无样本，跳过训练。")
        return None

    train_size = int(0.5 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"📊 {subset_name} 数据划分: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}")

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    dataset = QuantizedGraphDataset(df, lmdb_path=lmdb_path, use_lmdb=True)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: x)

    X_train, y_train = collect_data_from_dataloader(train_loader, "收集训练")
    if X_train is None or y_train is None:
        print("⚠️ 训练数据为空，跳过该子集。")
        return None

    X_val, y_val = collect_data_from_dataloader(val_loader, "收集验证")

    print("📌 开始训练 XGBoost 模型...")
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
        num_class=len(np.unique(y_train)),
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        max_depth=6
    )

    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        print(f"✅ 验证准确率: {val_acc:.4f}")
        # 在验证集上运行SHAP分析
        # 替换原来的SHAP分析
        gain_save_path = os.path.join(save_dir, f"{subset_name}_val_gain")
        analyze_feature_importance_gain(
            model=model,
            feature_names=FEATNAMES,
            save_path=gain_save_path
        )
    else:
        val_acc = None
        print("⚠️ 验证集为空。")
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{subset_name}_xgboost_model.pkl")
    history_path = os.path.join(save_dir, f"{subset_name}_training_history.json")

    pickle.dump(model, open(model_path, 'wb'))
    print(f"✅ 模型已保存为 '{model_path}'")

    training_history = {
        'final_val_accuracy': val_acc,
        'num_classes': len(np.unique(y_train)),
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0] if X_val is not None else 0,
        'classes': sorted(np.unique(y_train).tolist()),
        'subset': subset_name,
        'total_samples': len(indices)
    }
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"🎉 子集 {subset_name} 训练完成！\n")
    
    # 执行交叉测试
    cross_test_results = {}
    other_clusters = [cl for cl in all_clusters if cl != subset_name]
    
    for test_cluster in other_clusters:
        result = cross_test_model(
            model=model,
            train_cluster=subset_name,
            test_cluster=test_cluster,
            df=df,
            lmdb_path=lmdb_path,  # 使用统一的LMDB路径
            results_dir=save_dir
        )
        if result:
            cross_test_results[test_cluster] = result
    
    # 保存交叉测试汇总
    cross_summary_path = os.path.join(save_dir, f"{subset_name}_cross_test_summary.json")
    with open(cross_summary_path, 'w') as f:
        json.dump(cross_test_results, f, indent=2)
    
    print(f"📊 {subset_name} 的交叉测试汇总已保存为 '{cross_summary_path}'")
    
    return model, cross_test_results


# ----------------------------
# 生成交叉测试矩阵
# ----------------------------

def generate_cross_test_matrix(results_dir, clusters):
    """生成交叉测试准确率矩阵"""
    matrix = np.zeros((len(clusters), len(clusters)))
    matrix.fill(np.nan)
    
    for i, train_cluster in enumerate(clusters):
        summary_file = os.path.join(results_dir, f"{train_cluster}_cross_test_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            for j, test_cluster in enumerate(clusters):
                if train_cluster == test_cluster:
                    # 使用自身的验证准确率
                    history_file = os.path.join(results_dir, f"{train_cluster}_training_history.json")
                    if os.path.exists(history_file):
                        with open(history_file, 'r') as hf:
                            history = json.load(hf)
                            matrix[i, j] = history.get('final_val_accuracy', np.nan)
                else:
                    # 使用交叉测试准确率
                    if test_cluster in summary:
                        matrix[i, j] = summary[test_cluster].get('accuracy', np.nan)
    
    # 保存矩阵
    matrix_df = pd.DataFrame(matrix, index=clusters, columns=clusters)
    matrix_file = os.path.join(results_dir, "cross_test_accuracy_matrix.csv")
    matrix_df.to_csv(matrix_file)
    
    print(f"\n📊 交叉测试准确率矩阵已保存为 '{matrix_file}'")
    print("\n交叉测试准确率矩阵:")
    print(matrix_df.round(4))
    
    return matrix_df


# ----------------------------
# 主程序入口
# ----------------------------

if __name__ == "__main__":
    #  import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)  # 避免 fork 导致的问题
    rfam_df = pd.read_csv('rfam_pdb_labeled.csv', sep='\t')

    subsets_to_train = [
        "CL00111", "CL00112", "CL00001", "CL00113",
        "CL00009", "CL00006", "CL00123", "CL00102"
    ]

    # subsets_to_train = [
    #     "CL00123", "CL00102"
    # ]
    
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # 使用统一的LMDB路径
    lmdb_path = './rfam_lmdb'

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # dataset = QuantizedGraphDataset(rfam_df[rfam_df['CL'] == "CL00123"], lmdb_path=lmdb_path, use_lmdb=True)
    
    # 创建"All"类别数据集
    all_indices = create_all_cluster_dataset(rfam_df, sampling_ratio=0.5)
    
    # # 训练所有子集并进行交叉测试
    # trained_models = {}
    # all_cross_results = {}

    # # 先训练"All"类别作为baseline
    # print("\n" + "="*60)
    # print("🎯 首先训练 'All' 类别 (Baseline)")
    # print("="*60)
    
    # model_and_results = train_model_on_subset(
    #     df=rfam_df,
    #     subset_name="All",
    #     lmdb_path=lmdb_path,
    #     save_dir=results_dir,
    #     all_clusters=subsets_to_train + ["All"],
    #     indices=all_indices  # 提供自定义indices
    # )
    
    # if model_and_results:
    #     trained_models["All"] = model_and_results[0]
    #     all_cross_results["All"] = model_and_results[1]

    # # 然后训练各个cluster
    # print("\n" + "="*60)
    # print("🎯 训练各个Clusters")
    # print("="*60)
    
    # for cl in subsets_to_train:
    #     model_and_results = train_model_on_subset(
    #         df=rfam_df,
    #         subset_name=cl,
    #         lmdb_path=lmdb_path,
    #         save_dir=results_dir,
    #         all_clusters=subsets_to_train + ["All"]  # 包含"All"在交叉测试中
    #     )
        
    #     if model_and_results:
    #         trained_models[cl] = model_and_results[0]
    #         all_cross_results[cl] = model_and_results[1]

    # # 生成交叉测试矩阵（包含All类别）
    # print("\n" + "="*60)
    # print("📊 生成交叉测试准确率矩阵")
    # print("="*60)
    
    # all_clusters_for_matrix = ["All"] + subsets_to_train
    # generate_cross_test_matrix(results_dir, all_clusters_for_matrix)
    
    # print("\n🎉 所有训练和交叉测试完成！")
    
    all_clusters_for_analysis = ["All"] + subsets_to_train
    summarize_gain_importance("results", all_clusters_for_analysis, FEATNAMES, top_n=50)
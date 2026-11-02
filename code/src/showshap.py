import numpy as np
import matplotlib.pyplot as plt
import shap

# ----------------------------
# 特征名
# ----------------------------

NODEFEAT_NAMES = ['bond_P-O5', 'bond_O5-C5', 'bond_C5-C4', 'bond_C4-C3', 'bond_C3-O3', 'bond_O3-N', 'dist_P-C5', 'dist_P-C4', 'dist_O5-C4', 'dist_O5-C3', 'dist_C5-C3', 'dist_C5-O3', 'dist_C4-O3', 'dist_C4-N', 'dist_C3-N', 'dist_P-P_shift1', 'dist_P-P_shift2', 'dist_P-P_shift3', 'dist_P-O5_shift1', 'dist_P-O5_shift2', 'dist_P-O5_shift3', 'dist_C4-C4_shift1', 'dist_C4-C4_shift2', 'dist_C4-C4_shift3', 'dist_C4-P_shift1', 'dist_C4-P_shift2', 'dist_C4-P_shift3', 'angle_P_P_P_span1', 'angle_P_P_P_span2', 'angle_P_P_P_span3', 'angle_P_C4_P_span1', 'angle_P_C4_P_span2', 'angle_P_C4_P_span3', 'angle_C4_C4_C4_span1', 'angle_C4_C4_C4_span2', 'angle_C4_C4_C4_span3', 'sugar_planarity', 'delta_P_x', 'delta_P_y', 'delta_P_z', 'delta2_P_x', 'delta2_P_y', 'delta2_P_z', 'delta_O5_x', 'delta_O5_y', 'delta_O5_z', 'delta2_O5_x', 'delta2_O5_y', 'delta2_O5_z', 'delta_C5_x', 'delta_C5_y', 'delta_C5_z', 'delta2_C5_x', 'delta2_C5_y', 'delta2_C5_z', 'delta_C4_x', 'delta_C4_y', 'delta_C4_z', 'delta2_C4_x', 'delta2_C4_y', 'delta2_C4_z', 'delta_C3_x', 'delta_C3_y', 'delta_C3_z', 'delta2_C3_x', 'delta2_C3_y', 'delta2_C3_z', 'delta_O3_x', 'delta_O3_y', 'delta_O3_z', 'delta2_O3_x', 'delta2_O3_y', 'delta2_O3_z', 'delta_N_x', 'delta_N_y', 'delta_N_z', 'delta2_N_x', 'delta2_N_y', 'delta2_N_z', 'center_x', 'center_y', 'center_z', 'p_dist_mean', 'p_dist_std', 'p_dist_skew', 'p_dist_kurtosis', 'P_kappa_curvature', 'O5_kappa_curvature', 'C5_kappa_curvature', 'C4_kappa_curvature', 'C3_kappa_curvature', 'O3_kappa_curvature', 'N_kappa_curvature', 'dihedral_sin', 'dihedral_cos', 'p_window_std', 'sasa_4A', 'vec_p_center_x', 'vec_p_center_y', 'vec_p_center_z', 'inertia_min', 'inertia_max', 'inertia_ratio', 'dist_global_ctr', 'dist_main_dim', 'puckering_0', 'puckering_1', 'helical_rise', 'helical_twist', 'helical_roll', 'helical_tilt', 'C4_local_curvature_5', 'sequential_turn_span_angle', 'stacking', 'hydration', 'backbone_dir_cons5', 'backbone_dir_cons11', 'elec_cluster6', 'elec_cluster15', 'stack_density5', 'stack_density11', 'dyn_betw10', 'dyn_betw21', 'local_helix5', 'local_helix11', 'torsion_var5', 'torsion_var7', 'ring_distort', 'base_orient', 'P_kappa', 'P_tau', 'C4_kappa', 'C4_tau', 'O5_kappa', 'O5_tau', 'local_density', 'contact_num', 'ss_tendency', 'tri_P_O3prev_O5', 'tri_P_O5_C5', 'tri_C4_C5_C3', 'tri_C4_C3_O3', 'tri_tau_P_C4_next', 'va_P_O5_vs_C5_C4', 'va_base_vs_backbone', 'pp_dir_cons_3', 'ribose_plane_cons_3', 'base_stack_cons_3', 'bond_PO_cv_3', 'radius_cv_3', 'lambda2_std_3', 'tilt_sigma_3', 'dm_frob_cons_3', 'tor_smooth_3', 'pp_dir_cons_5', 'ribose_plane_cons_5', 'base_stack_cons_5', 'bond_PO_cv_5', 'radius_cv_5', 'lambda2_std_5', 'tilt_sigma_5', 'dm_frob_cons_5', 'tor_smooth_5', 'pp_dir_cons_7', 'ribose_plane_cons_7', 'base_stack_cons_7', 'bond_PO_cv_7', 'radius_cv_7', 'lambda2_std_7', 'tilt_sigma_7', 'dm_frob_cons_7', 'tor_smooth_7', 'pp_dir_cons_11', 'ribose_plane_cons_11', 'base_stack_cons_11', 'bond_PO_cv_11', 'radius_cv_11', 'lambda2_std_11', 'tilt_sigma_11', 'dm_frob_cons_11', 'tor_smooth_11', 'pp_dir_cons_13', 'ribose_plane_cons_13', 'base_stack_cons_13', 'bond_PO_cv_13', 'radius_cv_13', 'lambda2_std_13', 'tilt_sigma_13', 'dm_frob_cons_13', 'tor_smooth_13']

EDGEFEAT_NAMES = ['dist_P-P_rbf0', 'dist_P-P_rbf1', 'dist_P-P_rbf2', 'dist_P-P_rbf3', 'dist_P-P_rbf4', 'dist_P-P_rbf5', 'dist_P-P_rbf6', 'dist_P-P_rbf7', 'dist_P-P_rbf8', 'dist_P-P_rbf9', 'dist_P-P_rbf10', 'dist_P-P_rbf11', 'dist_P-P_rbf12', 'dist_P-P_rbf13', 'dist_P-P_rbf14', 'dist_P-P_rbf15', 'dist_P-P_rbf16', 'dist_P-P_rbf17', 'dist_P-P_rbf18', 'dist_P-P_rbf19', 'dist_P-P_rbf20', 'dist_P-P_rbf21', 'dist_P-P_rbf22', 'dist_P-P_rbf23', 'dist_P-P_rbf24', 'dist_P-P_rbf25', 'dist_P-P_rbf26', 'dist_P-P_rbf27', 'dist_P-P_rbf28', 'dist_P-P_rbf29', 'dist_P-P_rbf30', 'dist_P-P_rbf31', 'dist_O5-P_rbf0', 'dist_O5-P_rbf1', 'dist_O5-P_rbf2', 'dist_O5-P_rbf3', 'dist_O5-P_rbf4', 'dist_O5-P_rbf5', 'dist_O5-P_rbf6', 'dist_O5-P_rbf7', 'dist_O5-P_rbf8', 'dist_O5-P_rbf9', 'dist_O5-P_rbf10', 'dist_O5-P_rbf11', 'dist_O5-P_rbf12', 'dist_O5-P_rbf13', 'dist_O5-P_rbf14', 'dist_O5-P_rbf15', 'dist_O5-P_rbf16', 'dist_O5-P_rbf17', 'dist_O5-P_rbf18', 'dist_O5-P_rbf19', 'dist_O5-P_rbf20', 'dist_O5-P_rbf21', 'dist_O5-P_rbf22', 'dist_O5-P_rbf23', 'dist_O5-P_rbf24', 'dist_O5-P_rbf25', 'dist_O5-P_rbf26', 'dist_O5-P_rbf27', 'dist_O5-P_rbf28', 'dist_O5-P_rbf29', 'dist_O5-P_rbf30', 'dist_O5-P_rbf31', 'dist_C5-P_rbf0', 'dist_C5-P_rbf1', 'dist_C5-P_rbf2', 'dist_C5-P_rbf3', 'dist_C5-P_rbf4', 'dist_C5-P_rbf5', 'dist_C5-P_rbf6', 'dist_C5-P_rbf7', 'dist_C5-P_rbf8', 'dist_C5-P_rbf9', 'dist_C5-P_rbf10', 'dist_C5-P_rbf11', 'dist_C5-P_rbf12', 'dist_C5-P_rbf13', 'dist_C5-P_rbf14', 'dist_C5-P_rbf15', 'dist_C5-P_rbf16', 'dist_C5-P_rbf17', 'dist_C5-P_rbf18', 'dist_C5-P_rbf19', 'dist_C5-P_rbf20', 'dist_C5-P_rbf21', 'dist_C5-P_rbf22', 'dist_C5-P_rbf23', 'dist_C5-P_rbf24', 'dist_C5-P_rbf25', 'dist_C5-P_rbf26', 'dist_C5-P_rbf27', 'dist_C5-P_rbf28', 'dist_C5-P_rbf29', 'dist_C5-P_rbf30', 'dist_C5-P_rbf31', 'dist_C4-P_rbf0', 'dist_C4-P_rbf1', 'dist_C4-P_rbf2', 'dist_C4-P_rbf3', 'dist_C4-P_rbf4', 'dist_C4-P_rbf5', 'dist_C4-P_rbf6', 'dist_C4-P_rbf7', 'dist_C4-P_rbf8', 'dist_C4-P_rbf9', 'dist_C4-P_rbf10', 'dist_C4-P_rbf11', 'dist_C4-P_rbf12', 'dist_C4-P_rbf13', 'dist_C4-P_rbf14', 'dist_C4-P_rbf15', 'dist_C4-P_rbf16', 'dist_C4-P_rbf17', 'dist_C4-P_rbf18', 'dist_C4-P_rbf19', 'dist_C4-P_rbf20', 'dist_C4-P_rbf21', 'dist_C4-P_rbf22', 'dist_C4-P_rbf23', 'dist_C4-P_rbf24', 'dist_C4-P_rbf25', 'dist_C4-P_rbf26', 'dist_C4-P_rbf27', 'dist_C4-P_rbf28', 'dist_C4-P_rbf29', 'dist_C4-P_rbf30', 'dist_C4-P_rbf31', 'dist_C3-P_rbf0', 'dist_C3-P_rbf1', 'dist_C3-P_rbf2', 'dist_C3-P_rbf3', 'dist_C3-P_rbf4', 'dist_C3-P_rbf5', 'dist_C3-P_rbf6', 'dist_C3-P_rbf7', 'dist_C3-P_rbf8', 'dist_C3-P_rbf9', 'dist_C3-P_rbf10', 'dist_C3-P_rbf11', 'dist_C3-P_rbf12', 'dist_C3-P_rbf13', 'dist_C3-P_rbf14', 'dist_C3-P_rbf15', 'dist_C3-P_rbf16', 'dist_C3-P_rbf17', 'dist_C3-P_rbf18', 'dist_C3-P_rbf19', 'dist_C3-P_rbf20', 'dist_C3-P_rbf21', 'dist_C3-P_rbf22', 'dist_C3-P_rbf23', 'dist_C3-P_rbf24', 'dist_C3-P_rbf25', 'dist_C3-P_rbf26', 'dist_C3-P_rbf27', 'dist_C3-P_rbf28', 'dist_C3-P_rbf29', 'dist_C3-P_rbf30', 'dist_C3-P_rbf31', 'dist_O3-P_rbf0', 'dist_O3-P_rbf1', 'dist_O3-P_rbf2', 'dist_O3-P_rbf3', 'dist_O3-P_rbf4', 'dist_O3-P_rbf5', 'dist_O3-P_rbf6', 'dist_O3-P_rbf7', 'dist_O3-P_rbf8', 'dist_O3-P_rbf9', 'dist_O3-P_rbf10', 'dist_O3-P_rbf11', 'dist_O3-P_rbf12', 'dist_O3-P_rbf13', 'dist_O3-P_rbf14', 'dist_O3-P_rbf15', 'dist_O3-P_rbf16', 'dist_O3-P_rbf17', 'dist_O3-P_rbf18', 'dist_O3-P_rbf19', 'dist_O3-P_rbf20', 'dist_O3-P_rbf21', 'dist_O3-P_rbf22', 'dist_O3-P_rbf23', 'dist_O3-P_rbf24', 'dist_O3-P_rbf25', 'dist_O3-P_rbf26', 'dist_O3-P_rbf27', 'dist_O3-P_rbf28', 'dist_O3-P_rbf29', 'dist_O3-P_rbf30', 'dist_O3-P_rbf31', 'dist_P-N_rbf0', 'dist_P-N_rbf1', 'dist_P-N_rbf2', 'dist_P-N_rbf3', 'dist_P-N_rbf4', 'dist_P-N_rbf5', 'dist_P-N_rbf6', 'dist_P-N_rbf7', 'dist_P-N_rbf8', 'dist_P-N_rbf9', 'dist_P-N_rbf10', 'dist_P-N_rbf11', 'dist_P-N_rbf12', 'dist_P-N_rbf13', 'dist_P-N_rbf14', 'dist_P-N_rbf15', 'dist_P-N_rbf16', 'dist_P-N_rbf17', 'dist_P-N_rbf18', 'dist_P-N_rbf19', 'dist_P-N_rbf20', 'dist_P-N_rbf21', 'dist_P-N_rbf22', 'dist_P-N_rbf23', 'dist_P-N_rbf24', 'dist_P-N_rbf25', 'dist_P-N_rbf26', 'dist_P-N_rbf27', 'dist_P-N_rbf28', 'dist_P-N_rbf29', 'dist_P-N_rbf30', 'dist_P-N_rbf31', 'direct_x', 'direct_y', 'direct_z', 'quat_0', 'quat_1', 'quat_2', 'quat_3', 'pe_0', 'pe_1', 'pe_2', 'pe_3', 'pe_4', 'pe_5', 'pe_6', 'pe_7', 'pe_8', 'pe_9', 'pe_10', 'pe_11', 'pe_12', 'pe_13', 'pe_14', 'pe_15', 'pe_16', 'pe_17', 'pe_18', 'pe_19', 'pe_20', 'pe_21', 'pe_22', 'pe_23', 'pe_24', 'pe_25', 'pe_26', 'pe_27', 'pe_28', 'pe_29', 'pe_30', 'pe_31', 'rosetta_0', 'rosetta_1', 'rosetta_2', 'rosetta_3', 'rosetta_4', 'rosetta_5', 'rosetta_6', 'rosetta_7', 'rosetta_8', 'rosetta_9', 'rosetta_10', 'rosetta_11', 'rosetta_12', 'rosetta_13', 'rosetta_14', 'rosetta_15', 'rosetta_16', 'rosetta_17', 'rosetta_18', 'rosetta_19', 'rosetta_20', 'rosetta_21', 'rosetta_22', 'rosetta_23', 'rosetta_24', 'rosetta_25', 'rosetta_26', 'rosetta_27', 'rosetta_28', 'rosetta_29', 'rosetta_30', 'rosetta_31', 'rosetta_32', 'rosetta_33', 'rosetta_34', 'rosetta_35', 'rosetta_36', 'rosetta_37', 'rosetta_38', 'rosetta_39', 'cec_0', 'cec_1', 'cec_2', 'cec_3', 'cec_4', 'cec_5', 'cec_6', 'cec_7', 'pair_angle_1', 'pair_angle_2', 'pair_angle_3', 'pair_angle_4', 'dih_0_0', 'dih_0_1', 'dih_1_0', 'dih_1_1', 'dih_2_0', 'dih_2_1', 'dih_3_0', 'dih_3_1', 'dih_4_0', 'dih_4_1', 'dih_5_0', 'dih_5_1', 'dih_6_0', 'dih_6_1', 'tri_0_0', 'tri_0_1', 'tri_1_0', 'tri_1_1', 'tri_2_0', 'tri_2_1', 'tri_3_0', 'tri_3_1', 'tri_4_0', 'tri_4_1', 'tri_5_0', 'tri_5_1', 'tri_6_0', 'tri_6_1', 'stack_flag']


FEATNAMES = NODEFEAT_NAMES + [f'edge_{name}_mean' for name in EDGEFEAT_NAMES] + [f'edge_{name}_std' for name in EDGEFEAT_NAMES]


# 加载数据
shap_file = "./results/CL00111_outside_shap.npz"
shap_data = np.load(shap_file, allow_pickle=True)
print("Keys in SHAP file:", shap_data.keys())  # ['shap_values', 'expected_value']

shap_values = shap_data['shap_values']  # (10000, 878, 4)
expected_value = shap_data['expected_value']  # (4,) or scalar? check shape

print("SHAP values shape:", shap_values.shape)
print("Expected value:", expected_value)

# ========================
# 方法选择：多分类问题，我们通常按类别分析
# ========================

# 假设我们要分析第 0 类（可以根据需要更改）
class_idx = 0

# 提取该类的 SHAP 值: shape (10000, 878)
shap_values_class = shap_values[:, :, class_idx]

# 构造 Explanation 对象（用于新版本 SHAP 可视化）
# 注意：如果你没有原始特征数据，可以用零矩阵代替（仅用于绘图）
# 这里假设你知道特征名字，或用 generic 名字

feature_names = FEATNAMES

# 创建 Explanation 对象（推荐方式）
expl = shap.Explanation(
    values=shap_values_class,
    base_values=expected_value[class_idx],  # 每个样本的基础值（如果是全局标量，也可广播）
    data=np.zeros_like(shap_values_class),  # 占位符
    feature_names=feature_names
)

# ========================
# 1. Summary Plot（特征重要性）
# ========================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_class, features=expl.data, feature_names=expl.feature_names, show=False)
plt.title(f"SHAP Summary Plot - Class {class_idx}")
plt.tight_layout()
plt.savefig(f"./results/shap_summary_class_{class_idx}.png", dpi=300)
plt.show()

# ========================
# 2. 条形图：平均绝对 SHAP 值（全局特征重要性）
# ========================
plt.figure(figsize=(8, 6))
shap_importance = np.abs(shap_values_class).mean(axis=0)
idxs = np.argsort(shap_importance)[:20]  # top 20 最重要的特征
plt.barh(range(len(idxs)), shap_importance[idxs])
plt.yticks(range(len(idxs)), [feature_names[i] for i in idxs])
plt.xlabel("Mean |SHAP Value|")
plt.title(f"Top 20 Feature Importance (Class {class_idx})")
plt.tight_layout()
plt.savefig(f"./results/shap_importance_class_{class_idx}.png", dpi=300)
plt.show()

# ========================
# 3. 单样本 Force Plot（前两个样本示例）
# ========================
for i in range(2):
    shap.plots.force(
        expl.base_values,
        expl.values[i],
        expl.data[i],
        feature_names=expl.feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"Sample {i} - Class {class_idx} SHAP Force Plot")
    plt.tight_layout()
    plt.savefig(f"./results/shap_force_sample_{i}_class_{class_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ========================
# 4. 决策图（Decision Plot） - 展示前1000个样本的模型决策路径
# ========================
# 随机选择 1000 个样本
indices = np.random.choice(len(shap_values_class), size=1000, replace=False)
shap.decision_plot(
    base_value=expl.base_values,
    shap_values=shap_values_class[indices],
    features=expl.data[indices],
    feature_names=expl.feature_names,
    show=False
)
plt.title(f"SHAP Decision Plot (Class {class_idx}, 50 random samples)")
plt.tight_layout()
plt.savefig(f"./results/shap_decision_class_{class_idx}.png", dpi=300)
plt.show()
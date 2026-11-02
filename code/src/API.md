# RNKDB Model API Reference

面向 AI Agent 的模型接口文档。所有脚本从 `RNKDB/` 目录运行（如 `python ../src/train_on_rfam.py`）。

---

## 模块总览

```
src/
├── process_data.py              # PDB下载与RNA坐标提取
├── generate_torch_features_v0.py # 核心特征工程（PyTorch Geometric版图构建）
├── run_xgb_model_v2.py          # 特征工程（NumPy版）+ trRosetta边特征 + 统计特征
├── rfam_featurizer.py           # 批量图构建处理器
├── rfam_featurizer_quant.py     # 8-bit量化器
├── train_on_rfam.py             # XGBoost训练流水线（LMDB缓存 + 交叉验证）
├── showshap.py                  # SHAP值可视化
├── chemical.py                  # 化学定义（原子映射、残基类型）
├── cluster_by_seqsim.py         # CD-HIT聚类解析
├── tm_score_align.py            # TM-score结构比对
├── read_npy.py                  # C3_coords → PDB 转换
├── write_seq_for_cdhit.py       # 合并FASTA供CD-HIT输入
├── read_similar_train_test.py   # 训练/测试集结构相似性分析
└── config.py                    # 占位配置
```

---

## 1. process_data.py — 数据获取与处理

### `get_rna_seq_and_coords(pdb_id: str) -> tuple[str | None, np.ndarray | None, list[str] | None]`

从PDB文件中提取RNA序列和粗粒化坐标。

**输入:** PDB ID（如 `"3CCU"`），需先下载到 `./raw_pdb/`。

**返回:**
- `seq`: RNA序列字符串（如 `"AUCG..."`）
- `coords`: `(N, 7, 3)` numpy数组，7个骨架原子按序为 `P, O5', C5', C4', C3', O3', N1/N9`
- `chain_ids`: 每个残基对应的链ID列表

失败时返回 `(None, None, None)`。

**原子映射** 定义在函数内的 `aa2longalt` 字典，覆盖 RNA（A/C/G/U/RX）和 DNA（DA/DC/DG/DT/DX）。缺失原子填入 `NaN`。

### `process_single_pdb(pdb_id: str) -> None`

处理单个PDB：提取序列+坐标，按链拆分保存到 `./seq/<pdb>_<chain>.fasta` 和 `./coords/<pdb>_<chain>.npy`。

链ID前缀规则: 大写→`H_`，小写→`L_`，其他→`X_`。

### `load_and_filter(csv_file: str, pdb_out_dir: str, n_jobs: int = 4) -> None`

从NAKB的 select.csv 中筛选含RNA的复合物，并行下载PDB文件。

---

## 2. generate_torch_features_v0.py — 图特征构建（PyTorch版）

### `build_graph(coord: np.ndarray | torch.Tensor) -> dict`

**核心函数。** 从 `(N, 7, 3)` 坐标构建图。

**输入:** `coord` shape `(N, 7, 3)`，7个骨架原子坐标。

**返回字典:**
| Key | Shape | 说明 |
|-----|-------|------|
| `x` | `(N, ~130)` | 节点特征（键长、角度、曲率、统计量等） |
| `node_feat_name` | `list[str]` | 节点特征名列表，长度≈130 |
| `edge_index` | `(2, E)` | PyTorch Geometric格式边索引（k=32 KNN图） |
| `edge_attr` | `(E, ~400)` | 边特征（RBF距离、方向、四元数、位置编码、Rosetta特征等） |
| `edge_feat_name` | `list[str]` | 边特征名列表，长度≈400 |
| `graph_attr` | `dict` | 全局图特征（转动惯量、全局质心距等） |

**内部调用链:**
```
build_graph(coord)
  ├── knn_graph(coord.mean(dim=1), k=32)           # 构建KNN图
  ├── RNAEdgeFeatures.build(coord, edge_index)     # 边特征
  ├── extract_residue_features(coord)              # 基础残基特征
  ├── RNAGeometry.extract_enrich_features(coord)   # 几何增强特征
  ├── RNAGeometry.extract_angle_features(coord)    # 角度特征
  ├── RNAGeometry.extract_rolling_features(coord)  # 滑动窗口特征
  └── RNAGraphGlobal.global_shape_stats(coord)     # 全局形状特征
```

### `class DataConfig`

超参配置: `k_neighbors=32`, `coord_fill_null=True`, `seq_vocab="AUCG"`, `rbf_dim=32`

### `class BasicStats`

几何/统计算子集合（均为 `@staticmethod`）:

| 方法 | 签名 | 功能 |
|------|------|------|
| `normalize` | `(x: Tensor, dim=-1) -> Tensor` | L2归一化 |
| `distance` | `(a: Tensor, b: Tensor) -> Tensor` | 欧氏距离 |
| `rbf` | `(d: Tensor, num_rbf=32) -> Tensor` | RBF展开（中心0-30，宽1.5） |
| `angle` | `(a, b, c: Tensor) -> Tensor` | 三点角度（弧度） |
| `dihedral` | `(p0, p1, p2, p3: Tensor) -> Tensor` | 二面角 sin/cos |

### `class RNAEdgeFeatures`

构建边特征。核心方法:
- `build(coord, edge_index) -> tuple[Tensor, list[str]]`: 返回 `(edge_feats, edge_feat_names)`

### `class RNAGeometry`

残基级几何特征提取（静态方法集）:
- `extract_enrich_features(coord) -> dict`: 增强几何特征
- `extract_angle_features(coord) -> dict`: 角度/二面角特征
- `extract_rolling_features(coord) -> dict`: 滑动窗口统计特征
- `frenet_curvature_torsion(P) -> ...`: Frenet曲率/挠率

### `class RNAGraphGlobal`

全局图级特征:
- `global_shape_stats(coord) -> dict`: 转动惯量、全局质心距离等

---

## 3. run_xgb_model_v2.py — 特征工程（NumPy版）

### `build_graph(coord: np.ndarray) -> pd.DataFrame`

**备用图构建函数。** 纯NumPy实现，返回 `(N, D)` DataFrame，每行=一个残基的特征向量。

**输入:** `coord` shape `(N, 7, 3)`

**返回:** DataFrame，列包含节点特征 + 聚合后的边特征（mean/std）。

**内部调用:** `extract_statistical_features(coord, edge_index)` → 字典 → DataFrame

**与 `generate_torch_features_v0.build_graph` 的区别:** 此版本返回 DataFrame（便于直接喂给 XGBoost），不保留图结构。训练流水线实际使用的是 PyTorch 版的 build_graph。

### `extract_statistical_features(coord: np.ndarray, edge_index: np.ndarray) -> tuple[dict, np.ndarray]`

提取所有统计/几何特征。

**返回:** `(features_dict, edge_features_array)`
- `features_dict`: 键=特征名，值= `(L,)` 数组，包含：键长/角度/差分/曲率/二面角/全局统计
- `edge_features_array`: `(E, D)` 边特征

### 其他工具函数

| 函数 | 签名 | 功能 |
|------|------|------|
| `aggregate_features_mean` | `(edge_feats, targets, num_nodes) -> np.ndarray` | 边特征→节点 mean 聚合 |
| `aggregate_features_std` | `(edge_feats, targets, num_nodes) -> np.ndarray` | 边特征→节点 std 聚合 |
| `trRosetta_features` | `(coord, edge_index) -> np.ndarray` | trRosetta风格几何特征 `(E, 40)` |
| `edge_orient_features` | `(coords, edge_index) -> tuple` | 边方向向量 + 四元数 |
| `edge_dihedral` | `(coord, edge_index) -> np.ndarray` | 边二面角 sin/cos |
| `edge_trihedral` | `(coord, edge_index) -> np.ndarray` | 边三面角 sin/cos |
| `pair_angle` | `(coord, edge_index, k_shift=1) -> np.ndarray` | 边对角度 |

---

## 4. rfam_featurizer.py — 批量特征化

### `process_single_sample(args, coords_root_dir, seq_root_dir, output_dir, ...) -> dict`

处理单个PDB链样本。

**输入:** `args = (idx, row)`，row来自 `rfam_pdb_labeled.csv`（含 `pdb_id`, `chain` 列）

**流程:**
1. 加载 `coords/<pdb>_<chain>.npy`
2. 检查序列长度（5 < N < 500）
3. 调用 `generate_torch_features_v0.build_graph()`
4. 保存 `.npz` 到 `output_dir/proccessed/`

**返回:** `{'pdb_id', 'chain', 'status': 'success'|'error', 'error': ...}`

### `batch_process_quantization(rfam_df, coords_root_dir, seq_root_dir, output_dir, max_workers=4, ...) -> list`

批量并行处理全部样本。使用 `multiprocessing.Pool`。自动跳过已处理文件，重新处理损坏文件。

---

## 5. rfam_featurizer_quant.py — 特征量化

### `quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> tuple[Tensor, float, float, float]`

线性量化为 uint8。

**返回:** `(quantized_tensor, min_val, max_val, scale)`

### `dequantize_tensor(quantized_tensor, min_val, max_val, num_bits=8) -> Tensor`

还原量化张量。
```
dequantized = quantized * scale + min_val
```

### `calculate_error_metrics(original, restored) -> dict`

计算 MAE, MSE, RMSE, MaxError, SNR。

---

## 6. train_on_rfam.py — 训练流水线

### `class QuantizedGraphDataset(Dataset)`

PyTorch Dataset，加载量化图数据，支持LMDB缓存。

**构造:** `QuantizedGraphDataset(df, root_dir="rfam_quantization/quantized/", lmdb_path='./lmdb_data', use_lmdb=True)`

**数据加载流程:**
1. `__getitem__` → 从LMDB读取pickle → 去量化特征(`_dequantize_features`) → 聚合边特征(mean/std) → 拼接节点+边聚合特征 → 返回 `{'features': (N, D), 'labels': (N,), 'pdb_head': str}`
2. 首次运行自动创建LMDB缓存（`process_and_store_to_lmdb`）

**标签映射:** `{'A':0, 'C':1, 'G':2, 'U':3}`

**特征拼接:** 最终特征向量 = `x (节点特征) || edge_mean (边特征均值聚合) || edge_std (边特征标准差聚合)`，维度 ≈130 + 400 + 400 ≈ 930

### `collect_data_from_dataloader(dataloader, desc, expected_feature_dim=None) -> tuple[np.ndarray, np.ndarray]`

遍历DataLoader收集特征和标签，返回 `(X, y)` 可直接喂给 sklearn/XGBoost。

### `train_model_on_subset(df, subset_name, lmdb_path, save_dir, all_clusters, indices=None) -> tuple[XGBClassifier, dict] | None`

在单个Rfam cluster上训练XGBoost，并进行跨cluster交叉测试。

**模型配置:**
```python
XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=4,           # A/C/G/U
    n_estimators=100,
    max_depth=6,
    device='cuda',         # GPU if available
    tree_method='gpu_hist'
)
```

**输出文件（存于 `save_dir/`）:**
- `<subset>_xgboost_model.pkl` — 训练好的模型
- `<subset>_training_history.json` — 训练元信息
- `<subset>_val_gain_gain_importance.csv` — Gain特征重要性
- `<subset>_on_<other>_cross_test.json` — 交叉测试报告
- `<subset>_cross_test_summary.json` — 交叉测试汇总

### `cross_test_model(model, train_cluster, test_cluster, df, lmdb_path, results_dir) -> dict`

用训练好的模型在另一cluster上评估。

### `summarize_gain_importance(results_dir, clusters, feature_names, top_n=50) -> pd.DataFrame`

汇总所有交叉测试的Gain重要性，生成热图。

### `generate_cross_test_matrix(results_dir, clusters) -> pd.DataFrame`

生成 `(N_clusters × N_clusters)` 交叉测试准确率矩阵。

### 主程序入口

```python
# subsets_to_train 硬编码于文件末尾
subsets_to_train = ["CL00111", "CL00112", "CL00001", "CL00113",
                    "CL00009", "CL00006", "CL00123", "CL00102"]
```

运行前需修改此列表。

### `FEATNAMES` / `NODEFEAT_NAMES` / `EDGEFEAT_NAMES`

全局变量，定义所有特征名称。`FEATNAMES` = 节点特征 + 边特征mean + 边特征std。

---

## 7. chemical.py — 化学定义

无函数/类，纯数据定义模块:

| 变量 | 说明 |
|------|------|
| `num2aa` | 32元素列表：20种氨基酸 + UNK/MAS + 4种DNA + 4种RNA + N |
| `aa2num` | `num2aa` 的反向映射字典 |
| `aa2long` | 32×36 原子名元组（全原子表示） |
| `PDB_CHAIN_IDS` | 合法链ID字符集 |
| `NHEAVY=23`, `NTOTAL=36` | 重原子/总原子数 |
| `NAATOKENS=32` | 残基类型数 |

`process_data.py` 中另有独立的 `aa2longalt` 字典（粗粒化骨架原子映射）。

---

## 8. showshap.py — SHAP可视化

加载已保存的 SHAP `.npz` 文件并生成可视化：
- `shap.summary_plot()` — 特征重要性汇总
- `shap.decision_plot()` — 决策路径
- `shap.dependence_plot()` — 单特征依赖

**使用前** 需修改脚本内的 `shap_file` 路径指向实际的SHAP结果文件。

---

## 9. 辅助脚本

### `cluster_by_seqsim.py`
解析 CD-HIT 输出 (`.clstr`) → `cluster_by_seqsim_0.8.tsv`。

### `tm_score_align.py`
并行计算 TM-score 结构相似性，输出 `tmscore_result_between_train_test2.txt`。需要 `./TMscore` 二进制文件。

### `read_npy.py`
将 `C3_coords/*.npy` 转为最小PDB格式（仅C3'原子），用于结构比对。

### `write_seq_for_cdhit.py`
合并 `seq/` 中所有 FASTA → `for_cd_hit.fa`。

### `read_similar_train_test.py`
分析 TM-score 结果，筛选 TM-score ≥ 0.8 的样本对，写入 `test_id.csv`。

---

## 典型调用流程

```bash
# 1. 从PDB提取RNA序列和坐标
cd RNKDB
python -c "import sys; sys.path.insert(0,'../src'); from process_data import process_single_pdb; process_single_pdb('3CCU')"

# 2. 构建图特征（单样本）
python -c "
import sys; sys.path.insert(0,'../src')
import numpy as np
from generate_torch_features_v0 import build_graph
coord = np.load('coords/3CCU_H_A.npy')
graph = build_graph(np.nan_to_num(coord[:, :7]))
print(graph['x'].shape, graph['edge_attr'].shape)
"

# 3. 批量特征化
python ../src/rfam_featurizer.py

# 4. 训练（需先编辑 subsets_to_train）
python ../src/train_on_rfam.py

# 5. SHAP可视化
python ../src/showshap.py
```

## 依赖

```
torch, torch_geometric, xgboost, lightgbm, shap, biopython
pandas, numpy, scipy, scikit-learn, tqdm, lmdb, joblib, matplotlib, seaborn
```

环境管理: conda（见 `.vscode/settings.json`）。无 `requirements.txt`。

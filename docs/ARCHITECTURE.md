# SoupSeg: ssDNA-Informed Iterative Cell Segmentation for Stereo-seq

> **Stereo-seq v1.3 细胞分割工具：图像初始化 + 转录本约束迭代优化**

---

## 1. 项目概述

### 1.1 目标
开发一款专门针对 **Stereo-seq v1.3** 数据的细胞分割工具，核心思路：
1. **ssDNA 图像**提供细胞核形态约束（初始化）
2. **转录本空间分布**提供分子信息约束（迭代优化）
3. 两者联合迭代优化，输出精准细胞边界 + 转录本归属

### 1.2 为什么做这个
- **CellBin**：ssDNA 为主，转录本仅辅助验证（不迭代）
- **Baysor**：转录本为主，图像仅作为 prior（非 Stereo-seq 专用）
- **Gap**：没有一款成熟工具真正实现"图像初始化 + 转录本约束迭代联合优化"

### 1.3 核心优势
- 专为 Stereo-seq v1.3 0.5 μm 高分辨率设计
- 原生支持 ssDNA 图像 + 转录本多模态输入
- 显式建模 transcript diffusion
- 模块化设计，可单独使用各阶段

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     SoupSeg Pipeline                        │
└─────────────────────────────────────────────────────────────┘

输入层
──────────────────────────────────────────────────────────────
┌──────────────┐     ┌─────────────────────────────────────┐
│  ssDNA 图像   │     │      Stereo-seq 转录本数据          │
│  (TIFF/PNG)   │     │      (.bin / .coord / CSV)         │
└──────┬───────┘     └──────────────┬──────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────────────────────────────────────────────────┐
│            Stage 1: 图像预处理 (Image Preprocessing)       │
│  • 去噪 (CLAHE / Gaussian)                                │
│  • 对比度增强                                             │
│  • 背景估计与去除                                         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│           Stage 2: 核检测与初始化 (Nuclei Detection)       │
│  • 阈值分割 (Otsu / Adaptive)                            │
│  • 形态学操作 (Opening / Closing / Watershed)            │
│  • 核中心定位 (Centroid)                                  │
│  • 细胞边界初始化 (Nuclear Expansion)                     │
│  • ▼ v1.1.0: 自适应膨胀半径 (Adaptive Radius)             │
│    - 基于局部细胞密度、图像强度、转录本密度                 │
│    - 高密度区域 → 较小膨胀 | 低密度 → 较大膨胀            │
│  输出: 初始细胞多边形边界 + 核掩膜                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│         Stage 3: 转录本归属 (Transcript Assignment)       │
│  • Voronoi 图初始化细胞归属                               │
│  • 基于距离的初始分配                                      │
│  • 基因共表达相似性校正                                   │
│  • 计算每个细胞的转录本数量分布                            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│       Stage 4: 迭代优化 (Iterative Refinement)           │
│  • 边界能量计算 (图像梯度 + 转录本密度)                    │
│  • ▼ v1.1.0: U-Net 边界检测 (PyTorch)                   │
│    - 4-level U-Net, 输入 ssDNA → 边界概率热力图           │
│    - BCE + Dice Loss, 支持 tiling inference               │
│  • ▼ v1.1.0: GNN 边界细化 (PyTorch GAT)                  │
│    - 细胞图 (nodes=细胞, edges=邻接)                      │
│    - 2-layer Graph Attention Network                      │
│  • Graph Cut / Random Walk 优化                           │
│  • 扩散校正 (Deconvolution)                               │
│  • 异常细胞检测与修正                                      │
│  • 迭代收敛判断                                           │
│  输出: 最终细胞边界 + 转录本归属                           │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
输出层
──────────────────────────────────────────────────────────────
    • 细胞分割结果 (CSV/JSON/GeoJSON)
    • 每个细胞的转录本列表
    • 质量报告 (QC metrics)
    • 可视化叠加图
```

---

## 3. 核心算法设计

### 3.1 Stage 1: 图像预处理

```python
def preprocess_image(ssdna_img, method='clahe'):
    """
    输入: ssDNA 图像 (2D numpy array)
    输出: 预处理后的图像
    
    步骤:
    1. 高斯去噪 (sigma=1-2)
    2. CLAHE 对比度增强 (clip_limit=0.03)
    3. 背景估计 (Rolling ball / 高斯模糊差分)
    4. 归一化 [0, 1]
    """
```

**参数说明：**
- `clip_limit`: CLAHE 对比度限制，建议 0.02-0.05
- `sigma`: 高斯滤波核大小，根据图像质量调整
- `block_size`: 自适应阈值块大小（核分割用）

### 3.2 Stage 2: 核检测与边界初始化

```python
def detect_nuclei_and_initialize(preprocessed_img, expansion_radius_um=6.0):
    """
    核心思想: 基于 ssDNA 的核染色强度，定位细胞核，
             然后膨胀到合理细胞边界作为初始化
    
    输入: 预处理后的 ssDNA 图像
    输出: 
        - nuclei_mask: 二值核掩膜
        - cell_polygons: GeoJSON 格式的细胞边界
        - cell_centers: 每个细胞的中心坐标
    
    算法步骤:
    1. Otsu / 自适应阈值 → 二值化
    2. 形态学开操作 → 去除噪点
    3. 连通域分析 → 获取独立核区域
    4. 形态学膨胀 → 膨胀到细胞边界 (expansion_radius)
    5. 过滤过小/过大的区域 → 细胞大小先验过滤
    """
```

**关键参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `expansion_radius_um` | 6.0 μm | 核膨胀半径，需根据组织类型调整 |
| `min_nuclei_size_um2` | 50 μm² | 最小核面积过滤 |
| `max_nuclei_size_um2` | 2000 μm² | 最大核面积过滤 |
| `threshold_method` | 'otsu' | 阈值方法 |

### 3.3 Stage 3: 转录本归属

```python
def assign_transcripts(transcripts_df, cell_polygons, method='voronoi'):
    """
    将转录本分配到对应的细胞
    
    输入:
        transcripts_df: DataFrame with ['x', 'y', 'gene'] columns
        cell_polygons: GeoJSON polygons
    
    方法:
        1. Voronoi: 计算每个转录本属于哪个细胞（基于距离）
        2. Gene-coexpression: 同一细胞的转录本基因表达应相似
        3. Diffusion correction: 考虑扩散，同基因转录本可能来自多个细胞
    
    输出:
        每个转录本增加了 'cell_id' 列
    """
```

### 3.4 Stage 4: 迭代优化（核心创新）

```python
def iterative_refinement(cell_polygons, transcripts_df, max_iter=20, tolerance=0.001):
    """
    核心迭代优化循环
    
    能量函数:
    E = α * E_image + β * E_transcript + γ * E_prior
    
    其中:
    - E_image: 图像能量（边界是否沿图像梯度）
    - E_transcript: 转录本能量（同细胞转录本是否空间聚集）
    - E_prior: 先验能量（细胞大小、形状是否合理）
    
    优化算法:
    1. 计算当前边界上的能量梯度
    2. 使用贪婪/梯度下降更新边界位置
    3. 重新计算转录本归属
    4. 检查收敛 (|E_new - E_old| < tolerance)
    """
    
    for iteration in range(max_iter):
        # Step 1: 基于当前边界，计算转录本归属
        new_assignments = update_transcript_assignment(cell_polygons, transcripts_df)
        
        # Step 2: 计算边界能量
        energy = compute_boundary_energy(cell_polygons, new_assignments)
        
        # Step 3: 边界优化（Graph Cut 或 Random Walk）
        cell_polygons = optimize_boundaries(cell_polygons, new_assignments, energy)
        
        # Step 4: 收敛检查
        if abs(energy - prev_energy) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
    
    return cell_polygons, final_assignments
```

---

## 4. 数据结构

### 4.1 输入数据格式

**ssDNA 图像：**
- 格式: TIFF (16-bit) 或 PNG (8-bit)
- 分辨率: 0.5 μm/pixel (Stereo-seq v1.3)
- 推荐尺寸: 2048x2048 起

**Stereo-seq 转录本：**
```csv
# 转录本坐标文件 (transcripts.csv)
x,y,gene,count
1234.5,5678.2,ACTB,1
1235.1,5679.0,ACTB,2
...
```

### 4.2 输出数据格式

**细胞分割结果：**
```json
// cell_segmentation_result.json
{
  "metadata": {
    "source": "SoupSeg v0.1",
    "stereo_seq_version": "v1.3",
    "n_cells": 15234,
    "n_transcripts": 8934521
  },
  "cells": [
    {
      "cell_id": "cell_00001",
      "polygon": [[x1,y1], [x2,y2], ...],  // GeoJSON 格式
      "centroid": [x, y],
      "area_um2": 345.6,
      "n_transcripts": 523,
      "n_genes": 312,
      "top_genes": ["ACTB", "GAPDH", "VIM"]
    },
    ...
  ],
  "transcripts": [
    {
      "transcript_id": "tx_000001",
      "x": 1234.5,
      "y": 5678.2,
      "gene": "ACTB",
      "cell_id": "cell_00001"
    },
    ...
  ]
}
```

---

## 5. 模块依赖

```txt
# 核心依赖 (Python >= 3.8)
numpy>=1.21
scipy>=1.7
pandas>=1.3
scikit-image>=0.19      # 图像处理
scikit-learn>=1.0       # 聚类、度量
shapely>=2.0            # 几何操作
geopandas>=0.10         # GeoJSON 支持
networkx>=3.0           # 图操作
cv2 (opencv)>=4.5       # 图像处理
tifffile>=2021.7        # 读取 TIFF

# 分割相关
cellpose>=2.2           # 可选：核分割辅助
pydantic>=1.9           # 数据验证

# 加速（可选）
numba>=0.55             # JIT 编译加速
joblib>=1.1             # 并行处理

# 可视化
matplotlib>=3.5
seaborn>=0.11
scanpy>=1.9             # ST 数据分析
squidpy>=1.1            # ST 可视化

# 深度学习（可选，用于 learned segmentation）
torch>=1.10
torch-geometric>=2.0    # GNN（如果用 Bering-style 方法）

# 测试
pytest>=7.0
pytest-cov>=3.0
```

---

## 6. 配置参数

```yaml
# config/default.yaml

soupseg:
  # 数据路径
  input:
    ssdna_image: "data/ssdna.tiff"
    transcript_file: "data/transcripts.csv"
    registration_matrix: "data/reg_matrix.txt"  # 可选
  
  # Stereo-seq v1.3 参数
  stereo_seq:
    pixel_size_um: 0.5        # 分辨率
    bin_size: 1               # BIN 尺寸
  
  # Stage 1: 图像预处理
  preprocessing:
    gaussian_sigma: 1.5
    clahe_clip_limit: 0.03
    clahe_kernel_size: [8, 8]
  
  # Stage 2: 核检测
  nuclei_detection:
    method: "otsu"            # otsu / adaptive / unet
    min_area_um2: 50
    max_area_um2: 2000
    expansion_radius_um: 6.0  # 关键参数：核膨胀半径
  
  # Stage 3: 转录本归属
  transcript_assignment:
    method: "voronoi"          # voronoi / graph / bayesian
    use_gene_coexpression: true
    diffusion_model: "gaussian"  # gaussian / psf
  
  # Stage 4: 迭代优化
  refinement:
    max_iterations: 20
    tolerance: 0.001
    energy_weights:
      alpha_image: 0.3         # 图像梯度权重
      beta_transcript: 0.5     # 转录本约束权重
      gamma_prior: 0.2         # 先验权重
  
  # 输出
  output:
    format: "json"            # json / csv / geojson
    save_transcript_assignment: true
    save_visualization: true
    visualization_dpi: 300
```

---

## 7. 文件结构

```
soup-seg/
├── README.md
├── LICENSE
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-gpu.txt      # GPU 版本依赖
│
├── src/
│   └── soupseg/
│       ├── __init__.py
│       ├── version.py
│       │
│       ├── pipeline.py       # 主流程入口
│       │
│       ├── stages/
│       │   ├── __init__.py
│       │   ├── preprocess.py  # Stage 1: 图像预处理
│       │   ├── nuclei.py     # Stage 2: 核检测
│       │   ├── assignment.py  # Stage 3: 转录本归属
│       │   ├── refine.py     # Stage 4: 迭代优化
│       │   └── adaptive_radius.py  # v1.1.0: 自适应膨胀半径
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cell.py        # Cell 数据结构
│       │   ├── transcript.py # Transcript 数据结构
│       │   ├── unet_boundary.py  # v1.1.0: U-Net 边界检测
│       │   ├── gnn_boundary.py  # v1.1.0: GNN 边界细化
│       │   └── adaptive_radius.py  # v1.1.0: 自适应半径计算
│
├── config/
│   └── default.yaml          # 默认配置
│
├── docs/
│   ├── ARCHITECTURE.md       # 本文档
│   ├── API.md                # API 文档
│   ├── TUTORIAL.md           # 使用教程
│   └── BENCHMARK.md          # 基准测试方法
│
├── notebooks/
│   ├── 01_quickstart.ipynb   # 快速开始
│   ├── 02_stereoseq_v1.3.ipynb # Stereo-seq v1.3 教程
│   └── 03_benchmark.ipynb    # Benchmark 方法
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_nuclei.py
│   ├── test_assignment.py
│   ├── test_refine.py
│   └── test_pipeline.py
│
└── weights/                  # 模型权重（如用深度学习）
    └── README.md
```

---

## 8. 使用示例

### 8.1 快速使用

```python
from soupseg import SoupSeg

# 初始化
seg = SoupSeg(config='config/default.yaml')

# 运行完整流程
result = seg.run(
    ssdna_image='path/to/ssdna.tiff',
    transcript_file='path/to/transcripts.csv',
    output_dir='output/'
)

# 查看结果
print(f"分割了 {result.n_cells} 个细胞")
print(f"转录本归属率: {result.assignment_rate:.1%}")

# 可视化
result.plot_overlay(save='output/segmentation.png')
```

### 8.2 分步使用

```python
from soupseg.stages import preprocess, detect_nuclei, assign_transcripts, refine

# Step 1: 预处理
img = preprocess.load_image('ssdna.tiff')
preprocessed = preprocess.apply(img, clahe=True, denoise=True)

# Step 2: 核检测与初始化
nuclei_mask, cell_polygons = detect_nuclei(preprocessed, expansion_radius_um=6.0)

# Step 3: 转录本归属
transcripts = assign_transcripts.load('transcripts.csv')
assignments = assign_transcripts.voronoi(transcripts, cell_polygons)

# Step 4: 迭代优化
final_polygons, final_assignments = refine.iterative(
    cell_polygons, transcripts, assignments,
    max_iter=20
)

# 输出
result = refine.save(final_polygons, final_assignments, 'output/')
```

---

## 9. 已知限制与未来方向

### 9.1 当前版本限制
- 仅支持 Stereo-seq v1.3（其他版本需调整分辨率参数）
- 需要 ssDNA 图像与转录本坐标精确配准（配准误差 < 1 μm）
- 大视野数据（> 1 cm²）计算量大，建议分块处理
- 仅支持 2D 切片，暂不支持 3D 重建

### 9.2 计划开发
- [x] v1.1.0: U-Net / GNN 用于边界检测 (PyTorch)
- [x] v1.1.0: 基于数据自适应调整膨胀半径
- [ ] 3D 支持：从连续切片重建 3D 细胞结构
- [ ] GPU 加速：CuPy / PyTorch 实现
- [ ] Web 界面：基于 napari 的可视化编辑

---

## 10. 参考文献

1. Petukhov et al. (2022). Cell segmentation in imaging-based spatial transcriptomics. *Nature Biotechnology*. PMID: 34650268
2. Ren et al. (2025). Systematic benchmarking of Stereo-seq. *Nature Communications*. PMID: 41107232
3. Baysor (Nolan lab). Bayesian segmentation of spatial transcriptomics. GitHub: nolanlab/baysor
4. CellBin (BGI). Stereo-seq official analysis pipeline.
5. Bering (2025). Graph deep learning for joint segmentation. *Nature Communications*. PMID: 40681510

---

*Document Version: 0.1 | Date: 2026-04-23*

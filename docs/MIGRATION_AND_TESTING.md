# SoupSeg A30 服务器迁移与测试指南

**用户指南**：gateswell  
**目标服务器**：A30 GPU 服务器  
**日期**：2026-04-23  

---

## 1. 概览

本文档指导你在 A30 GPU 服务器上完成 SoupSeg 的环境配置、数据准备、测试运行和结果反馈。

```
开发环境 (Raspberry Pi)          测试环境 (A30 GPU 服务器)
┌─────────────────────┐        ┌─────────────────────────┐
│ 项目代码已推送到      │        │ Clone 代码               │
│ GitHub               │───►    │ 安装依赖                 │
│ gateswell/SOUP-seg  │        │ 准备 Stereo-seq 数据     │
└─────────────────────┘        │ 运行测试                 │
                                 │ 反馈结果                │
                                 └─────────────────────────┘
```

---

## 2. 第一步：服务器环境准备

### 2.1 Clone 代码

```bash
# SSH 登录 A30 服务器后
cd /path/to/your/workdir
git clone https://github.com/gateswell/SOUP-seg.git
cd SOUP-seg
```

### 2.2 创建 Python 环境（推荐 conda）

```bash
# 创建独立环境
conda create -n soupseg python=3.9 -y
conda activate soupseg

# 安装核心依赖
pip install -r requirements.txt

# 安装额外依赖（Graph Cut + 加速）
pip install PyMaxflow numba joblib

# 可选：GPU 加速（如果使用 CuPy）
pip install cupy-cuda11x  # 根据你的 CUDA 版本选择
```

### 2.3 验证安装

```bash
python -c "
import soupseg
from soupseg.stages import iterative_refinement, RefinementConfig
print('SoupSeg version:', soupseg.__version__)
print('All imports OK')
"
```

---

## 3. 第二步：准备测试数据

### 3.1 数据要求

| 数据类型 | 必需 | 格式 | 描述 |
|---------|------|------|------|
| ssDNA 图像 | ✅ | TIFF/PNG | 核染色图像，灰度或单通道 |
| 转录本坐标 | ✅ | CSV | 包含 x, y, gene 列 |
| 注册矩阵 | ❌ | JSON/NPY | 图像对齐变换矩阵 |

**ssDNA 图像规格**：
- 分辨率：0.5 μm/pixel（Stereo-seq v1.3）
- 推荐尺寸：2048×2048 或 4096×4096
- 格式：TIFF（16-bit 推荐）, PNG（8-bit）
- 命名建议：`ssdna_tile_{row}_{col}.tiff`

**转录本 CSV 格式**：
```csv
transcript_id,x,y,gene,count
tx_00000000,1523,2048,GAPDH,1
tx_00000001,1621,2091,ACTB,1
tx_00000002,1705,2156,EGFR,2
```
- `x, y`：像素坐标（从 0 开始）
- `gene`：基因名称
- `count`：UMI 计数（默认为 1）

### 3.2 生成 Mock 测试数据（无需真实数据）

如果你还没有真实数据，先用 Mock 数据验证流程：

```bash
# 进入项目目录
cd SOUP-seg

# 生成 Mock 数据
python tests/generate_mock_data.py \
    --output-dir tests/data/ \
    --image-size 1024 \
    --n-cells 20 \
    --tx-per-cell 50 \
    --n-background 200

# 查看生成的文件
ls tests/data/
# 应看到：mock_ssdna.tiff, mock_transcripts.csv, cell_ground_truth.json
```

### 3.3 数据目录结构（建议）

```
/path/to/stereoseq_data/
├── ssdna/
│   ├── sample_001_ssdna.tiff
│   ├── sample_002_ssdna.tiff
│   └── ...
├── transcripts/
│   ├── sample_001_transcripts.csv
│   ├── sample_002_transcripts.csv
│   └── ...
└── registration/
    ├── sample_001_matrix.json
    └── ...
```

---

## 4. 第三步：运行测试

### 4.1 快速测试（Mock 数据）

```bash
cd /path/to/SOUP-seg

# 方法一：直接运行 Python
python -c "
from soupseg import SoupSeg

seg = SoupSeg()
result = seg.run(
    ssdna_image='tests/data/mock_ssdna.tiff',
    transcript_file='tests/data/mock_transcripts.csv',
    output_dir='tests/output/',
    save_visualization=True
)
print(result.summary())
"

# 方法二：使用 pytest 测试
pytest tests/test_refine.py -v
```

### 4.2 真实数据测试

```bash
cd /path/to/SOUP-seg

python -c "
from soupseg import SoupSeg

seg = SoupSeg()

# 根据你的数据调整配置
seg.config['expansion_radius_um'] = 6.0  # 可调整
seg.config['refinement_config']['max_iterations'] = 20

result = seg.run(
    ssdna_image='/path/to/your/ssdna.tiff',
    transcript_file='/path/to/your/transcripts.csv',
    output_dir='/path/to/output/',
    save_visualization=True
)

print('=' * 50)
print('测试完成！')
print('=' * 50)
print(result.summary())
"
```

### 4.3 单独测试某个 Stage

```python
# 只测试 Stage 4（如果你已有初始分割结果）
from soupseg.stages import iterative_refinement, RefinementConfig
from skimage import filters
import numpy as np

# 加载数据
cell_mask = np.load('your_initial_mask.npy')
transcript_coords = np.load('your_transcripts.npy')  # Nx2 array

# 计算梯度
gradient = filters.sobel(your_image.astype(float))

# 运行 Stage 4
config = RefinementConfig(
    max_iterations=20,
    alpha_image=0.4,
    beta_transcript=0.4,
    verbose=True
)

refined_mask, props, info = iterative_refinement(
    cell_mask,
    gradient,
    transcript_coords,
    cell_properties=[],
    config=config
)

print(f'收敛: {info[\"converged\"]}')
print(f'迭代次数: {info[\"iterations\"]}')
print(f'最终能量: {info[\"final_energy\"]:.6f}')
```

---

## 5. 第四步：解读测试结果

### 5.1 输出文件

运行后在 `output_dir` 中会生成：

```
output/
├── cells.json              # 细胞分割结果 (GeoJSON)
├── transcripts.json        # 转录本归属结果
├── metadata.json           # 运行元数据
├── convergence.json        # Stage 4 收敛信息
└── visualization/          # 可视化图像（如果启用）
    ├── segmentation.png    # 分割结果叠加图
    ├── transcripts.png     # 转录本分布图
    └── refinement/         # 迭代过程（如果启用）
        ├── iter_01.png
        ├── iter_02.png
        └── ...
```

### 5.2 关键指标

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `n_cells_final` | 最终细胞数 | 取决于图像大小 |
| `assignment_rate` | 转录本归属率 | > 70% 为良好 |
| `convergence.iterations` | 实际迭代次数 | ≤ max_iterations |
| `convergence.converged` | 是否收敛 | True 为理想 |
| `convergence.final_energy` | 最终能量值 | 越小越好（相对） |

### 5.3 convergence.json 示例

```json
{
    "iterations": 12,
    "converged": true,
    "final_energy": 0.0234,
    "energies": [0.0456, 0.0389, 0.0312, ...],
    "energy_components": [
        {"E_image": 0.015, "E_transcript": 0.008, "E_prior": 0.001},
        ...
    ],
    "method": "graphcut"
}
```

---

## 6. 第五步：反馈结果给 AI

### 6.1 需要反馈的信息

测试完成后，请将以下信息发送给 AI（通过 OpenClaw）：

#### 必须提供：
1. **数据概况**
   - 图像尺寸（height × width）
   - 细胞数量（初始 / 最终）
   - 转录本总数

2. **运行结果**
   - 转录本归属率
   - Stage 4 收敛状态（converged: true/false）
   - 实际迭代次数 / 最大迭代次数
   - 最终能量值

3. **问题描述**（如果有）
   - 报错信息（粘贴完整 traceback）
   - 内存溢出（OOM）
   - 速度问题
   - 分割效果不佳

#### 可选提供：
- 可视化结果截图
- 期望与实际差异描述
- 特殊组织类型或数据特点

### 6.2 反馈模板

```markdown
## A30 测试反馈

### 数据
- 图像：xxx × xxx 像素
- 转录本数：xxxxx
- 组织类型：[简单描述]

### 结果
- 初始细胞数：xxx
- 最终细胞数：xxx
- 归属率：xx.x%
- 收敛：是/否
- 迭代次数：xx/20
- 最终能量：0.xxxxx

### 问题（如果有）
[粘贴报错或描述问题]

### 可视化
[可选：截图链接或描述]
```

### 6.3 示例反馈

```
## A30 测试反馈

### 数据
- 图像：4096 × 4096 像素
- 转录本数：45,231
- 组织类型：小鼠脑皮层

### 结果
- 初始细胞数：156
- 最终细胞数：151
- 归属率：87.3%
- 收敛：是
- 迭代次数：12/20
- 最终能量：0.0234

### 问题
无报错，但归属率比预期低（87% vs 期望 >90%）

### 观察
部分细胞边界在密集转录本区域有轻微重叠
```

---

## 7. 常见问题排查

### 7.1 PyMaxflow 安装失败

```bash
# 错误：ModuleNotFoundError: No module named 'maxflow'
# 解决：

pip install PyMaxflow  # 如果失败

# 或者使用 conda
conda install -c conda-forge py-maxflow

# 如果仍然失败，手动安装
git clone https://github.com/pmneila/PyMaxflow.git
cd PyMaxflow
python setup.py install
```

### 7.2 内存不足（OOM）

```python
# 解决：分块处理大图像

from soupseg.pipeline import SoupSeg

seg = SoupSeg()

# 方法一：缩小图像
# 在传入前将图像 resize 到 2048x2048

# 方法二：分 Tile 处理
# 将大图切成 2048x2048 的小块，分别处理后拼接

# 方法三：减少迭代次数
seg.config['refinement_config']['max_iterations'] = 10
```

### 7.3 转录本归属率低

可能原因：
1. 转录本与细胞核对齐不良 → 检查注册矩阵
2. 细胞膨胀半径过大 → 减小 `expansion_radius_um`
3. 细胞过于密集 → 调整 Voronoi 参数

```python
# 调整配置
seg.config['expansion_radius_um'] = 4.0  # 从 6.0 减小
seg.config['max_distance_um'] = 15.0      # 增加最大距离
```

### 7.4 Stage 4 不收敛

```python
# 调整能量权重
seg.config['refinement_config']['alpha_image'] = 0.5  # 增加图像权重
seg.config['refinement_config']['beta_transcript'] = 0.3  # 减小转录本权重
seg.config['refinement_config']['tolerance'] = 0.01  # 放宽收敛条件
```

---

## 8. 下一步优化方向

根据测试结果，AI 会针对性地优化：

| 症状 | 可能的优化方向 |
|------|--------------|
| 归属率低 | 改进扩散校正算法 |
| 边界不平滑 | 增加边界平滑项 |
| 计算太慢 | GPU 加速或分块处理 |
| 小细胞误检 | 调整面积阈值 |
| 迭代不收敛 | 调整能量函数权重 |

---

## 9. 联系方式

有问题随时反馈：
- 在 OpenClaw 中发送消息
- 或在 GitHub Issue 中描述问题：https://github.com/gateswell/SOUP-seg/issues

---

*最后更新：2026-04-23 23:30 GMT+8*

# 模型特征子空间分析工具 (Model Feature Subspace Analysis Tools)

本模块提供了一套完整的工具，用于对比分析多个SNN模型的特征子空间差异。

## 📁 文件结构

```
ESVAE/analysis/
├── __init__.py                    # 模块初始化
├── README.md                      # 本文档
├── utils.py                       # 共享工具函数
├── compare_parameters.py          # 脚本1: 参数对比
├── extract_features_tsne.py       # 脚本2: 特征提取与t-SNE可视化
└── compare_subspace_cka.py        # 脚本3: CKA/TCKA子空间对比
```

## 🎯 功能概述

### 1. `compare_parameters.py` - 模型参数对比

**功能：**
- 加载两个模型的checkpoint
- 逐层计算参数差异（L2范数、余弦相似度）
- 生成详细的对比报告（CSV格式）
- 重点分析关键层（dvs_input、features、bottleneck、classifier）

**使用方法：**

```bash
# 从项目根目录运行
python -m ESVAE.analysis.compare_parameters \
    --baseline_ckpt /path/to/baseline_model.pth \
    --finetuned_ckpt /path/to/pretrained_finetuned_model.pth \
    --output_dir results/parameter_comparison
```

**输出文件：**
- `parameter_comparison_full.csv` - 完整的参数对比结果
- `parameter_comparison_dvs_input.csv` - DVS输入层参数对比
- `parameter_comparison_features.csv` - 特征层参数对比
- `parameter_comparison_bottleneck.csv` - Bottleneck层参数对比
- `parameter_comparison_classifier.csv` - 分类器层参数对比

**关键指标：**
- **L2 Difference**: 参数向量的L2范数差异，越大表示参数变化越大
- **Cosine Similarity**: 参数向量的余弦相似度，越接近1表示方向越相似

---

### 2. `extract_features_tsne.py` - 特征提取与t-SNE可视化

**功能：**
- 从两个模型中提取DVS特征
- 使用t-SNE降维到2D空间
- 生成多种可视化对比图
- 直观展示特征空间的类簇结构

**使用方法：**

```bash
# 从项目根目录运行
python -m ESVAE.analysis.extract_features_tsne \
    --baseline_ckpt /path/to/baseline_model.pth \
    --pretrained_ckpt /path/to/pretrained_model.pth \
    --data_path /path/to/n-caltech101 \
    --output_dir results/tsne_visualization \
    --max_samples 2000 \
    --layer_name bottleneck
```

**参数说明：**
- `--max_samples`: 使用的样本数量（默认2000）
- `--layer_name`: 要提取特征的层名称（默认bottleneck）
- `--perplexity`: t-SNE的perplexity参数（默认30）
- `--learning_rate`: t-SNE的学习率（默认200）

**输出文件：**
- `tsne_baseline.png` - Baseline模型的t-SNE可视化
- `tsne_pretrained.png` - 预训练模型的t-SNE可视化
- `tsne_comparison.png` - 并排对比图
- `tsne_overlay.png` - 叠加显示图（使用不同marker）

**可视化说明：**
- 每个点代表一个样本
- 颜色代表类别
- 点的聚集程度反映类内紧密度
- 不同颜色点的分离程度反映类间可分性

---

### 3. `compare_subspace_cka.py` - CKA/TCKA子空间对比

**功能：**
- 提取时序特征（N, T, D格式）
- 计算多种CKA指标定量对比子空间
- 分析编码层和高层特征的相似度
- 生成详细的数值报告

**使用方法：**

```bash
# 从项目根目录运行
python -m ESVAE.analysis.compare_subspace_cka \
    --baseline_ckpt /path/to/baseline_model.pth \
    --pretrained_ckpt /path/to/pretrained_model.pth \
    --data_path /path/to/n-caltech101 \
    --output_dir results/cka_comparison \
    --max_samples 2000
```

**输出文件：**
- `cka_comparison_results.txt` - 详细的CKA对比结果

**CKA指标说明：**

1. **Temporal Linear CKA (TCKA)**
   - 对每个时间步分别计算CKA，然后取平均
   - 适用于SNN的时序特征对比
   - 公式：`TCKA = (1/T) * Σ CKA(f_t^A, f_t^B)`

2. **Linear CKA (SUM)**
   - 先对时间维度求和，再计算CKA
   - 关注整体时序信息的累积效果

3. **Linear CKA (FLATTEN)**
   - 将时间维度展平后计算CKA
   - 保留完整的时序模式信息

**CKA值解释：**
- CKA ∈ [0, 1]
- CKA = 1: 两个特征子空间完全相同
- CKA = 0: 两个特征子空间完全不相关
- CKA > 0.8: 高度相似
- 0.5 < CKA < 0.8: 中等相似
- CKA < 0.5: 相似度较低

---

## 🔧 工具函数 (`utils.py`)

### 主要功能：

1. **`load_model_checkpoint()`**
   - 加载模型checkpoint
   - 自动处理DataParallel格式
   - 支持多种checkpoint格式

2. **`FeatureExtractor`**
   - 使用hook机制提取中间层特征
   - 支持多层同时提取
   - 自动管理hook的注册和清理

3. **`extract_features_from_dataloader()`**
   - 批量提取特征
   - 自动处理时序维度
   - 支持设置最大样本数

4. **`get_layer_output_with_mem()`**
   - 专门用于SNN的时序特征提取
   - 保留完整的时间维度信息
   - 可选返回membrane potential

5. **`compute_parameter_difference()`**
   - 计算两个模型参数的L2差异和余弦相似度
   - 支持逐层对比

6. **`print_layer_names()`**
   - 打印模型所有层的名称
   - 用于调试和确定layer_name

---

## 📊 典型工作流程

### 场景：对比baseline和预训练+微调模型

```bash
# 步骤1: 对比模型参数
python -m ESVAE.analysis.compare_parameters \
    --baseline_ckpt checkpoints/baseline.pth \
    --finetuned_ckpt checkpoints/pretrained_finetuned.pth \
    --output_dir results/param_comparison

# 步骤2: 可视化特征空间
python -m ESVAE.analysis.extract_features_tsne \
    --baseline_ckpt checkpoints/baseline.pth \
    --pretrained_ckpt checkpoints/pretrained_finetuned.pth \
    --data_path /path/to/n-caltech101 \
    --output_dir results/tsne_vis \
    --max_samples 2000

# 步骤3: 定量对比子空间
python -m ESVAE.analysis.compare_subspace_cka \
    --baseline_ckpt checkpoints/baseline.pth \
    --pretrained_ckpt checkpoints/pretrained_finetuned.pth \
    --data_path /path/to/n-caltech101 \
    --output_dir results/cka_comparison \
    --max_samples 2000
```

---

## 🔍 如何选择layer_name

要查看模型中所有可用的层名称，可以使用以下代码：

```python
from ESVAE.models.snn_models.VGG import VGGSNN
from ESVAE.analysis.utils import print_layer_names

model = VGGSNN(in_channel=2, cls_num=101, img_shape=48)
print_layer_names(model)
```

**常用层名称：**
- `dvs_input` - DVS输入层（编码层）
- `features` - 特征提取层（整个Sequential）
- `features.0` - 第一个特征层
- `features.1` - 第二个特征层（池化后）
- `bottleneck` - 瓶颈层（高层特征）
- `classifier` - 分类器

---

## 📝 代码示例

### 示例1: 在Python脚本中使用

```python
from ESVAE.analysis.compare_parameters import compare_model_parameters

# 对比参数
results = compare_model_parameters(
    baseline_ckpt_path="checkpoints/baseline.pth",
    finetuned_ckpt_path="checkpoints/finetuned.pth",
    output_dir="results/comparison",
    device="cuda"
)

# 查看结果
print(results.head())
```

### 示例2: 自定义特征提取

```python
import torch
from ESVAE.models.snn_models.VGG import VGGSNN
from ESVAE.analysis.utils import FeatureExtractor

# 创建模型
model = VGGSNN(in_channel=2, cls_num=101, img_shape=48)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 创建特征提取器
extractor = FeatureExtractor(model, ['dvs_input', 'bottleneck'])

# 提取特征
features_dict = extractor.extract(input_data)

# 访问特征
dvs_features = features_dict['dvs_input']
bottleneck_features = features_dict['bottleneck']

# 清理
extractor.remove_hooks()
```

---

## ⚙️ 环境要求

```bash
# 必需的Python包
torch>=1.8.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

---

## 🐛 常见问题

### Q1: 找不到layer_name
**A:** 使用`print_layer_names()`查看所有可用层名称，确保拼写正确。

### Q2: CUDA out of memory
**A:** 减小`--batch_size`或`--max_samples`参数。

### Q3: t-SNE运行时间过长
**A:** 减小`--max_samples`（推荐1000-2000），或调整`--perplexity`参数。

### Q4: CKA值为NaN
**A:** 检查特征是否包含NaN或Inf，可能需要在模型训练时添加梯度裁剪。

### Q5: 特征维度不匹配
**A:** 确保两个模型使用相同的架构和输入尺寸。

---

## 📚 参考文献

1. **CKA (Centered Kernel Alignment)**
   - Kornblith et al. "Similarity of Neural Network Representations Revisited." ICML 2019.

2. **t-SNE**
   - van der Maaten & Hinton. "Visualizing Data using t-SNE." JMLR 2008.

3. **Temporal Efficient Training**
   - Deng et al. "Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting." ICLR 2022.

---

## 📧 联系方式

如有问题或建议，请联系项目维护者。

---

## 📄 许可证

本模块遵循项目主许可证。


# MetricANN

一个**模块化**、**可增量**、**少样本**的度量学习 + ANN（近似最近邻）检索框架，适用于海量类别（千级至百万级），每类仅需 1–3 张样本即可。支持动态更新，兼容 CPU、GPU 或分布式集群。

---

## 🔍 项目概览

MetricANN 实现了一个基于 embedding 的检索流水线：

1. **度量学习**：训练共享网络（例如 ResNet 骨干 + 投射头），使用三元组或对比损失  
2. **嵌入提取**：批量编码所有标注图像，生成固定长度向量  
3. **ANN 索引**：用 FAISS 或 HNSW 对向量进行子线性搜索索引  
4. **检索 CLI**：在线对查询图提取向量并检索 Top‑K 类别  
5. **增量更新**：可在不重训/不全量重建索引的前提下添加新类别或补充样本  
6. **分布式准备**：提供多节点推理的抽象与占位

---

## 🚀 功能与组件

| 层级           | 模块                          | 职责                                    |
| -------------- | ----------------------------- | --------------------------------------- |
| **配置**       | `config.py`                   | 全局常量、超参数、路径                  |
| **数据 I/O**   | `data_loader.py`              | 通用 Dataset + 自定义 collate_fn       |
|                | `triplet_dataset.py`          | 三元组采样器（anchor/positive/negative） |
| **模型**       | `model/embedding_model.py`    | 骨干→投射头→L2 归一化                   |
|                | `model/trainer.py`            | TripletTrainer（含 checkpoint、早停）   |
| **训练 CLI**   | `train.py`                    | 训练循环、tqdm 进度条、日志、早停        |
| **提取**       | `extract.py`                  | 导出每张图的 `.npy` 嵌入向量             |
| **索引**       | `indexer/`                    | `IndexBackend` + `FaissIndex`/`HNSWIndex` |
| **建索引脚本** | `build_index.py`              | 扫描 embeddings、自动调优 nlist、训练 & 保存 |
| **检索 CLI**   | `search.py`                   | 实时提取查询向量 → 搜索索引 → 输出 Top‑K |
| **增量**       | `incremental/updater.py`      | 加载索引 → 增量添加向量 → 重新保存       |
| **分布式**     | `distributed/client.py`       | 未来 RPC/微服务客户端占位               |
| **工具**       | `utils/logging.py`            | 统一日志配置                            |
|                | `utils/metrics.py`            | Top‑K 准确率、MRR 等指标                |
| **脚本**       | `run_train.py`/`run_train.bat`| 一键启动训练封装                        |
| **元信息**     | `requirements.txt`、`README.md` | 依赖列表、使用说明                       |

---

## 📂 目录结构
```

MetricANN/
 ├── README.md
 ├── requirements.txt
 ├── config.py
 ├── data_loader.py
 ├── triplet_dataset.py
 ├── train.py
 ├── extract.py
 ├── build_index.py
 ├── search.py
 ├── incremental/
 │   └── updater.py
 ├── distributed/
 │   └── client.py
 ├── indexer/
 │   ├── backend.py
 │   ├── faiss_index.py
 │   └── hnsw_index.py
 ├── model/
 │   ├── embedding_model.py
 │   └── trainer.py
 ├── utils/
 │   ├── logging.py
 │   └── metrics.py
 ├── run_train.py
 └── run_train.bat

```
---

## ⚙️ 安装

### 1. 创建 & 激活 Conda 环境

```bash
conda create -n metricann_env python=3.9 -y
conda activate metricann_env
```

### 2. 安装依赖

使用 **mamba**（速度更快）：

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --override-channels
mamba install -c conda-forge faiss-cpu hnswlib pillow numpy tqdm -y
```

或使用 **pip**（官方 GPU Wheel）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu hnswlib pillow numpy tqdm
```

------

## 🗂 数据准备

在项目根目录下创建 `dataset/`，结构示例：

```
dataset/
├── class_0001/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── …
├── class_0002/
│   └── …
└── …
```

- 每个子文件夹代表一个类别标签
- 建议每类至少 1–3 张样本，后续可补更多提高鲁棒性
- 可选地再拆分 `train/`、`val/` 以做早停与评估

------

## 🎯 训练

### （1）一键脚本

```bash
python run_train.py
```

### （2）CLI 方式

```bash
python train.py \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --val_data dataset/val \
  --patience 5 \
  --ckpt_dir checkpoints \
  --model_out final_model.pt
```

- **早停**：若验证集损失在 `--patience` 个 epoch 内无提升则停止
- **检查点**：每个 epoch 会产出 `checkpoints/ckpt_epoch{n}.pt` 与 `best_model.pt`
- **日志**：启动汇总、每 epoch 训练/验证损失、检查点路径

------

## 📈 嵌入提取

```bash
python extract.py --model final_model.pt --device cuda
```

在 `embeddings/` 下生成：

```
embeddings/
├── class_0001/001.npy
├── class_0001/002.npy
├── class_0002/…
└── …
```

------

## 🔍 构建 ANN 索引

```bash
python build_index.py
```

脚本会：

1. 读取所有 `.npy` 嵌入
2. 自动设置 `nlist = max(1, 总向量数 // 10)`
3. 训练 FAISS 或 HNSW 索引
4. 保存到 `index/main_index.idx` 与 `main_index.labels`

------

## 🛠 搜索

```bash
python search.py \
  --model final_model.pt \
  --index index/main_index \
  --image query.jpg \
  --k 5 \
  --device cuda
```

输出 Top‑K 最相似类别及距离分数。

------

## 🔄 增量更新

有新类别或新增样本时只需：

```bash
python incremental/updater.py \
  --model final_model.pt \
  --index index/main_index \
  --new_data dataset/ \
  --device cuda
```

即可在现有索引上追加新向量，无需全量重建。

------

## 🌐 分布式集成

- `distributed/client.py` 提供 RPC/微服务客户端占位
- 后续可部署多台索引服务器，实现分片 & 负载均衡

------

## 📋 指标与监控

使用 `utils/metrics.py` 计算：

- **Top‑K 准确率**
- **平均倒数排名（MRR）**

结合 `utils/logging.py` 的日志可接入监控平台。

------

## 🔄 下一步 & 最佳实践

- **超参调优**：学习率、margin、batch size、`nlist`/`nprobe`、`M`/`ef`
- **数据增强**：提升光照/遮挡/裁剪等强鲁棒性
- **单元测试**：覆盖数据加载、模型、索引、检索
- **基准评测**：在真实场景下测量延迟 & 召回率
- **分布式扩展**：实现索引分片 & RPC 服务
- **持续在线学习**：接入标注流，自动化增量更新

------

## 📄 许可证

MIT © 2025 FurryFans

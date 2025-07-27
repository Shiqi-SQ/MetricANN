# 毛绒玩偶识别器

一个高度模块化、可增量、少样本的实例级图像识别系统，用于“毛绒玩偶”识别。  
每个玩偶实例——具有独特的形状、颜色和花纹——都能在仅有 1–3 张参考图像的情况下，加入到可扩展至 10⁵–10⁶ 类的数据库中。  
本 README 旨在帮助开发者以及大型语言模型（如 ChatGPT）快速、全面地了解、使用和扩展此项目。

---

## 🔍 项目概览

- **目标**  
  构建一个基于 embedding 的检索系统，输入一张（或部分）玩偶照片，返回最可能的玩偶“ID”（名称）。

- **核心需求**  
  1. **少样本**：每个玩偶仅需 1–3 张标注图  
  2. **大规模**：初始类别数 ≈ 数百，扩展至 10⁵–10⁶  
  3. **增量学习**：可动态添加新玩偶，或为已有玩偶增加新图，无需整体重训  
  4. **鲁棒性**：支持光照变化、遮挡、局部视图（如仅识别头部）  
  5. **跨平台**：训练时可用 GPU，推理支持 CPU/GPU（单机或集群）  
  6. **解耦合**：各模块职责清晰——数据加载、模型、索引、检索、存储、更新、工具、分布式

---

## 🚀 功能与组件

| 层级         | 模块                          | 职责                                          |
| ------------ | ----------------------------- | --------------------------------------------- |
| **配置**     | `config.py`                   | 全局常量、超参、路径                          |
| **数据**     | `data_loader.py`              | `PlushieDataset` 与 `get_dataloader()`       |
| **存储**     | `storage/`                    | 抽象与具体存储后端（`LocalStorage`、`DBStorage`） |
| **模型**     | `model/`                      | `EmbeddingNet`（ResNet→ℓ₂归一化向量）+ `TripletTrainer` |
| **提取**     | `extract.py`                  | 批量提取并保存 embedding 为 `.npy` 文件        |
| **索引**     | `indexer/`                    | `IndexBackend` + `FaissIndex` / `HNSWIndex`  |
| **检索**     | `search.py`                   | 加载模型 & 索引 → 查询 → 输出 Top‑K 结果      |
| **增量**     | `incremental/updater.py`      | 加载索引 → 添加新向量 → 保存                  |
| **分布式**   | `distributed/client.py`       | 未来 RPC 集群交互占位                         |
| **工具**     | `utils/`                      | `logging.py`、`metrics.py`（Top‑K、mAP、MRR）  |
| **元信息**   | `requirements.txt`, **README.md** | 依赖列表、项目说明                          |

---

## 📂 目录结构

```
plushie\_recognizer/
├── README.md
├── requirements.txt
├── config.py
├── data\_loader.py
├── storage/
│   ├── backend.py
│   ├── local\_storage.py
│   └── db\_storage.py
├── model/
│   ├── embedding\_model.py
│   └── trainer.py
├── extract.py
├── indexer/
│   ├── backend.py
│   ├── faiss\_index.py
│   └── hnsw\_index.py
├── search.py
├── incremental/
│   └── updater.py
├── distributed/
│   └── client.py
└── utils/
├── logging.py
└── metrics.py
````

---

## ⚙️ 安装

```bash
git clone https://github.com/your-org/plushie_recognizer.git
cd plushie_recognizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

* **可选 GPU**：若需 GPU 加速，将 `faiss-cpu` 换成 `faiss-gpu`
* **数据库后端**：如使用 `DBStorage`，需安装 SQLite；亦可改用其它数据库

---

## 🎯 快速上手

1. **准备数据集**

   ```
   dataset/
     plushie_001/
       img1.jpg
       img2.jpg
     plushie_002/
       head1.png
       ...
   ```

2. **训练 embedding 模型**

   ```bash
   python - <<EOF
   from model.trainer import TripletTrainer
   from data_loader import get_dataloader
   import torch

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   trainer = TripletTrainer(device, margin=0.2)
   dataloader = get_dataloader()
   trainer.train(epochs=10, train_loader=dataloader)
   trainer.save('model.pt')
   EOF
   ```

3. **提取嵌入向量**

   ```bash
   python extract.py --model model.pt --device cuda
   ```

4. **构建索引**

   ```python
   # build_index.py
   from indexer.faiss_index import FaissIndex
   import numpy as np, os

   idx = FaissIndex(dim=512, nlist=1000, nprobe=10)
   for label in os.listdir('embeddings'):
       for fn in os.listdir(f'embeddings/{label}'):
           vec = np.load(f'embeddings/{label}/{fn}')
           idx.add(label, vec)
   idx.build()
   idx.save('index/main_index')
   ```

5. **执行检索**

   ```bash
   python search.py --model model.pt --index index/main_index --image query.jpg --k 5
   ```

6. **增量更新**

   ```bash
   python incremental/updater.py --model model.pt --index index/main_index --new_data new_samples/ --device cpu
   ```

---

## 💡 项目解读指南（面向 LLM）

* **整体流程**：

  1. **数据** → 按标签组织图像
  2. **模型**：训练统一 embedding 网络
  3. **提取**：批量计算图像向量
  4. **索引**：将向量加载到 FAISS/HNSW，实现近似检索
  5. **检索**：对查询图像提取向量 → 搜索索引 → 输出 Top‑K
  6. **增量**：无需重训，直接向索引追加向量

* **解耦设计**：各目录分别负责

  * `storage/`：I/O
  * `model/`：网络与训练
  * `extract.py`：批量提取
  * `indexer/`：多种检索后端
  * `search.py`：CLI 接口
  * `incremental/`：更新逻辑
  * `distributed/`：分布式占位

* **可扩展点**：

  * 修改 `config.py` 中 `INDEX_TYPE` 为 `"hnsw"` 切换索引
  * 新存储后端：继承 `StorageBackend`
  * 新模型骨干：在 `config.py` 指定 `MODEL_BACKBONE`
  * 增加或改造 CLI、集成微服务

---

## 🔄 后续规划 & 最佳实践

* 编写单元测试，覆盖数据加载、模型、索引、检索等模块
* 在真实数据上基准测试检索延迟与准确率
* 调优 FAISS/HNSW 参数（`nlist`、`M`、`ef`）以平衡速度与召回
* 实现 `distributed/client.py`，构建分布式索引服务
* 集成监控与日志（`utils/logging.py`、`utils/metrics.py`）
* 分析识别错误案例，持续迭代优化

---

## 📄 许可证

MIT © 2025 Furryfans
欢迎用于商业或研究，并根据需要修改扩展。

# Glasses Detection Based on CNN-SE Attention

## 1. 项目简介

本项目实现了一个基于深度学习的人脸眼镜识别模型，用于判断输入人脸图像中的人物是否佩戴眼镜。该任务属于二分类图像分类问题，类别包括：

```text
0: no_glasses  未佩戴眼镜
1: glasses     佩戴眼镜
```

项目主要流程包括数据集整理、图像预处理、模型训练、验证集评估、早停控制和测试集结果输出。训练过程中会根据验证集准确率保存效果最好的模型权重文件 `best_model.pth`。

---

## 2. 数据集说明

本项目使用 CelebA 人脸属性数据集。CelebA 包含大量人脸图像及对应的人脸属性标签，本文选取其中的 `Eyeglasses` 属性作为分类标签，将任务转化为“是否佩戴眼镜”的二分类问题。

`list_attr_celeba.csv` 中的标签含义为：

```text
Eyeglasses = 1   表示佩戴眼镜
Eyeglasses = -1  表示未佩戴眼镜
```

原始数据集主要包括：

```text
data/data/raw_kaggle/celeba-dataset/
├── img_align_celeba/
├── list_attr_celeba.csv
├── list_bbox_celeba.csv
├── list_eval_partition.csv
└── list_landmarks_align_celeba.csv
```

为了提高实验效率，本项目从原始 CelebA 数据集中整理出一个小规模眼镜二分类数据集：

```text
data/mini_celeba/
├── images/
├── labels.csv
├── class_to_idx.json
└── split_summary.csv
```

其中，`images/` 保存训练、验证和测试图像；`labels.csv` 保存图像名、类别标签和数据集划分信息；`class_to_idx.json` 保存类别名称与编号的对应关系；`split_summary.csv` 保存数据划分统计结果。

---

## 3. 数据量与划分方式

本实验使用 5000 张图像，并采用类别均衡采样，使佩戴眼镜和未佩戴眼镜两类样本数量一致。

数据集划分比例为：

```text
训练集 : 验证集 : 测试集 = 8 : 1 : 1
```

具体数量如下：

| 数据集   | 图像数量 | 用途          |
| ----- | ---: | ----------- |
| Train | 4000 | 用于模型参数训练    |
| Val   |  500 | 用于模型选择和早停判断 |
| Test  |  500 | 用于最终性能测试    |

训练集中两类样本数量为：

| 类别         |   数量 |
| ---------- | ---: |
| no_glasses | 2000 |
| glasses    | 2000 |

类别索引为：

```json
{
  "no_glasses": 0,
  "glasses": 1
}
```

---

## 4. 运行环境

本项目基于 Python 和 PyTorch 实现，主要依赖包括：

```text
Python 3.8
PyTorch
Torchvision
Pandas
NumPy
Pillow
Tqdm
```

实验设备配置如下：

| 项目         | 配置                      |
| ---------- | ----------------------- |
| GPU        | NVIDIA GeForce RTX 4090 |
| 训练框架       | PyTorch                 |
| 任务类型       | 二分类图像分类                 |
| 输入图像尺寸     | 224 × 224               |
| Batch Size | 64                      |
| 混合精度训练     | AMP                     |

运行时设备信息示例：

```text
使用设备：cuda
GPU：NVIDIA GeForce RTX 4090
```

---

## 5. 图像预处理

训练阶段对输入图像进行以下处理：

```text
Resize 到 224 × 224
随机水平翻转
颜色扰动 ColorJitter
转换为 Tensor
按照 ImageNet 均值和标准差归一化
```

验证集和测试集不使用随机增强，只进行尺寸缩放、Tensor 转换和归一化，以保证评估结果稳定。

归一化参数为：

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 6. 模型结构

本项目采用 CNN 与 SE 注意力机制相结合的二分类模型。CNN 用于提取图像中的局部视觉特征，例如人脸轮廓、眼部纹理、眼镜边框等；SE 注意力模块用于学习不同通道的重要性权重，使模型更加关注与眼镜识别相关的特征。

模型整体结构为：

```text
Input Image
    ↓
CNN Feature Extractor
    ↓
SE Attention Module
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
Output: no_glasses / glasses
```

CNN 主干可分为 4 个卷积特征提取阶段：

| 阶段           | 输出通道数 | 作用             |
| ------------ | ----: | -------------- |
| Conv Block 1 |    32 | 提取边缘、颜色和简单纹理特征 |
| Conv Block 2 |    64 | 提取局部纹理和眼部区域结构  |
| Conv Block 3 |   128 | 提取更复杂的人脸局部语义特征 |
| Conv Block 4 |   256 | 提取高层分类相关特征     |

SE 模块包括两个步骤：

```text
Squeeze：通过全局平均池化压缩每个通道的二维特征图；
Excitation：通过小型全连接网络学习通道权重，并将权重乘回原特征图。
```

在眼镜识别任务中，SE 注意力机制有助于增强眼镜边框、镜片反光和眼部纹理等相关特征，提高模型的判别能力。

---

## 7. 训练策略

### 7.1 损失函数

本项目使用交叉熵损失函数 `CrossEntropyLoss` 进行二分类训练。由于训练集采用类别均衡采样，两类样本数量相同，因此类别权重设置为：

```text
no_glasses: 1.0
glasses: 1.0
```

### 7.2 优化器

模型训练使用 AdamW 优化器。AdamW 在 Adam 的基础上改进了权重衰减方式，有助于缓解过拟合。

主要训练参数如下：

| 参数            |     数值 |
| ------------- | -----: |
| Optimizer     |  AdamW |
| Learning Rate |  0.001 |
| Weight Decay  | 0.0001 |
| Batch Size    |     64 |
| Epochs        |     50 |
| AMP           |   True |

### 7.3 学习率调度

本项目采用 `CosineAnnealingLR` 余弦退火学习率调度策略。该策略在训练初期使用较大学习率以加快收敛，后期逐渐降低学习率，使模型在较优区域附近稳定优化。

初始学习率为：

```text
lr = 0.001
```

### 7.4 早停策略

本项目使用 Early Stopping，监控指标为验证集准确率 `val_acc`。

早停规则为：

```text
如果当前 val_acc 高于历史最佳值，则保存 best_model.pth，并重置早停计数器；
如果当前 val_acc 没有提升，则早停计数器加 1；
如果连续 patience 个 epoch 没有提升，则提前停止训练。
```

当前设置为：

```text
patience = 5
monitor = val_acc
mode = max
```

---

## 8. 模型保存与结果文件

训练过程中会保存以下文件：

```text
runs/glasses_resnet18/
├── best_model.pth
├── last_model.pth
├── history.json
└── test_metrics.json
```

文件说明如下：

| 文件                | 说明               |
| ----------------- | ---------------- |
| best_model.pth    | 验证集准确率最高的模型权重    |
| last_model.pth    | 最后一个 epoch 的模型权重 |
| history.json      | 每一轮训练和验证指标       |
| test_metrics.json | 最终测试集结果          |

最终测试时加载 `best_model.pth`，而不是直接使用最后一轮模型，以减少过拟合对最终结果的影响。

如果训练脚本导出 CSV 文件，通常保存在：

```text
runs/glasses_resnet18/results/
├── training_history.csv
├── final_test_metrics.csv
└── final_summary.csv
```

---

## 9. 实验结果

训练过程中，每个 epoch 会输出训练集准确率、验证集整体准确率以及每个类别的准确率。

第 1 个 epoch 输出示例：

```text
Epoch [1/10] 结果：
  train_loss          = 0.7103
  train_acc           = 64.15%
  val_loss            = 1.0987
  val_acc             = 50.80%
  val_no_glasses_acc  = 2.40%
  val_glasses_acc     = 99.20%
  lr                  = 0.00097553
```

指标含义如下：

| 指标                 | 含义              |
| ------------------ | --------------- |
| train_acc          | 训练集整体准确率        |
| val_acc            | 验证集整体准确率        |
| val_no_glasses_acc | 验证集中未佩戴眼镜类别的准确率 |
| val_glasses_acc    | 验证集中佩戴眼镜类别的准确率  |

从初期结果可以看出，模型可能存在类别预测偏向。因此，除了整体准确率外，还需要关注每一类的分类准确率。

最终测试结果可根据 `test_metrics.json` 填写：

```text
test_acc            = 待填写
test_no_glasses_acc = 待填写
test_glasses_acc    = 待填写
best_val_acc        = 待填写
stopped_epoch       = 待填写
```

---

## 10. 运行方式

进入项目目录：

```powershell
cd D:\YYF\glasses_detection
```

激活 Conda 环境：

```powershell
conda activate sei
```

整理眼镜二分类数据集：

```powershell
python download_celeba.py --raw_root data\data\raw_kaggle\celeba-dataset --out_dir data\mini_celeba --mini_size 5000 --overwrite
```

开始训练：

```powershell
python train.py --data_dir data\mini_celeba --save_dir runs\glasses_resnet18 --model_name resnet18 --epochs 50 --batch_size 64 --lr 0.001 --num_workers 0 --amp --patience 5
```

短训练测试命令：

```powershell
python train.py --data_dir data\mini_celeba --save_dir runs\glasses_resnet18 --model_name resnet18 --epochs 10 --batch_size 64 --lr 0.001 --num_workers 0 --amp --patience 3
```

安装依赖：

```powershell
pip install -r requirements.txt
```

---

## 11. 项目文件结构

项目当前结构如下：

```text
glasses_detection/
├── data/
├── logs/
├── results/
├── runs/
│   └── glasses_resnet18/
├── dataset.py
├── download_celeba.py
├── model.py
├── predict.py
├── readme.md
├── requirements.txt
└── train.py
```

### 11.1 data/

`data/` 用于存放原始数据和整理后的训练数据。

```text
data/data/raw_kaggle/celeba-dataset/   原始 CelebA 数据集
data/mini_celeba/                      整理后的眼镜二分类数据集
```

整理后的数据包括：

```text
data/mini_celeba/
├── images/
├── labels.csv
├── class_to_idx.json
└── split_summary.csv
```

### 11.2 logs/

`logs/` 用于保存数据处理、训练和调试过程中的日志文件，例如 `download.log` 和 `run1.txt`，便于复盘实验过程和定位问题。

### 11.3 results/

`results/` 用于保存项目层面的实验汇总结果，例如最终指标表、可视化图像和多次实验对比结果。

### 11.4 runs/

`runs/` 用于保存单次训练实验的输出，包括模型权重、训练历史、测试指标和自动生成的结果文件。

```text
runs/glasses_resnet18/
├── best_model.pth
├── last_model.pth
├── history.json
├── test_metrics.json
└── results/
```

### 11.5 dataset.py

`dataset.py` 定义数据集读取逻辑，主要负责读取 `labels.csv`，根据 `split` 字段划分 train / val / test，并读取对应图像和标签。

### 11.6 download_celeba.py

`download_celeba.py` 用于从原始 CelebA 数据集中提取 `Eyeglasses` 属性，生成眼镜二分类小数据集，并保存 `labels.csv`、`class_to_idx.json` 和 `split_summary.csv`。

### 11.7 model.py

`model.py` 用于定义模型结构，包括 ResNet18 或自定义 CNN-SE 模型。模型输入为人脸图像，输出为：

```text
0: no_glasses
1: glasses
```

### 11.8 train.py

`train.py` 是模型训练主程序，负责加载数据、构建模型、设置损失函数与优化器、执行训练和验证、保存最佳模型、进行早停判断，并在测试集上评估模型。

当前训练策略包括：

```text
Optimizer: AdamW
Loss: CrossEntropyLoss
Scheduler: CosineAnnealingLR
Early Stopping: val_acc
AMP: True
```

### 11.9 predict.py

`predict.py` 用于加载训练好的 `best_model.pth`，对新的图片进行预测，并输出该图片属于 `glasses` 或 `no_glasses` 的类别和置信度。

### 11.10 requirements.txt

`requirements.txt` 记录项目依赖库，便于快速复现实验环境。

主要依赖包括：

```text
torch
torchvision
pandas
numpy
pillow
tqdm
```

---

## 12. 总结

本项目基于 CelebA 数据集构建眼镜佩戴识别模型，使用 CNN 提取人脸图像特征，并通过 SE 注意力机制增强关键通道特征表达。训练过程中采用 AdamW、余弦退火学习率调度、混合精度训练和早停策略，以提高训练效率并降低过拟合风险。

实验结果表明，卷积神经网络能够有效建模眼镜识别任务。但在训练初期，模型可能存在类别预测偏向，因此除了整体准确率外，还需要关注 `glasses` 和 `no_glasses` 两类的分类准确率。

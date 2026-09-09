# 数据中心服务器指标异常检测系统

本项目针对 SMD（Server Machine Dataset）中的 `machine-1-1` 服务器指标数据，
封装 KAN-AD 模型并提供离线异常检测和逐点流式异常检测接口。

当前仓库的重点是检测模块，不包含前端界面、评估页面或其他检测算法。

## 1. 项目目录说明

```text
./
├── data/
│   ├── train/machine-1-1.txt
│   ├── test/machine-1-1.txt
│   └── test_label/machine-1-1.txt
├── models/
│   ├── kan_ad_detector.py       # 我们实现的统一 KAN-AD 封装
│   ├── __init__.py
│   └── KAN-AD/                  # 开源 KAN-AD 源码
├── tests/
│   ├── test_kan_ad_detector.py  # 轻量单元测试
│   └── validate_smd.py          # 真实 SMD 数据验证
└── README.md
```

## 2. 开源代码与本项目实现的代码

### 2.1 开源 KAN-AD 代码

以下目录来自开源 KAN-AD 仓库：

```text
models/KAN-AD/
```

其中主要包括：

- `models/KAN-AD/kanad/kanad.py`：原始 KAN-AD 网络和 EasyTSAD 方法类；
- `models/KAN-AD/kanad/__init__.py`：原始包导出文件；
- `models/KAN-AD/kanad/config.toml`：原始实验配置；
- `models/KAN-AD/run_exp.py`：原始 EasyTSAD 实验入口；
- `models/KAN-AD/pyproject.toml`：原始项目依赖配置。

### 2.2 本项目实现的 detector

主要实现文件为：

```text
models/kan_ad_detector.py
```

`KANADDetector` 使用开源代码中的 `KANADModel` 网络结构，并自行提供适合本项目的
`fit / decision_function / predict` 统一接口和流式推理接口。由于原始 `KANAD` 类
依赖 EasyTSAD 的实验生命周期，并没有直接提供需求文档假设的 `fit/predict` 接口，
封装层对原始网络进行了训练适配。

封装层的主要职责包括：

- 将 38 维时间序列转换为“历史窗口预测下一点”样本；
- 只使用训练集拟合 `StandardScaler`；
- 只使用训练集预测误差计算异常阈值；
- 将原始网络输出恢复为 38 维预测结果；
- 提供离线检测和逐点实时检测；
- 检查输入维度、训练状态和数值有效性。

## 3. `KANADDetector` 接口

导入方式：

```python
from models.kan_ad_detector import KANADDetector
```

### 3.1 构造函数

```python
KANADDetector(
    window_size=10,
    n_harmonics=5,
    percentile=99.0,
    device="cpu",
    *,
    epochs=10,
    batch_size=1024,
    learning_rate=0.01,
)
```

参数说明：

| 参数 | 类型 | 说明 |
|---|---|---|
| `window_size` | `int` | 历史滑动窗口长度，必须为正整数。 |
| `n_harmonics` | `int` | KAN/Fourier 网络阶数，必须为正整数。 |
| `percentile` | `float` | 用训练集误差计算阈值的百分位数，范围 `0~100`。 |
| `device` | `str` | `"cpu"` 或 `"cuda"`。CUDA 不可用时回退到 CPU。 |
| `epochs` | `int` | 训练轮数，默认 `10`。 |
| `batch_size` | `int` | 训练和推理批大小，默认 `1024`。 |
| `learning_rate` | `float` | Adam 优化器学习率，默认 `0.01`。 |

输入数据统一为形状 `(n, 38)` 的数值数组。`38` 对应 SMD 服务器的 38 个指标。

### 3.2 `fit(X_train) -> None`

使用正常训练集拟合 detector。

处理步骤：

1. 检查 `X_train` 是否为 `(n, 38)` 且只包含有限数值；
2. 在训练集上拟合 `StandardScaler`；
3. 构造长度为 `window_size` 的历史窗口和下一时刻目标；
4. 训练 KAN-AD 网络；
5. 在训练集上计算预测均方误差；
6. 根据 `percentile` 计算并固定异常阈值 `threshold_`。

返回值为 `None`。训练数据长度必须大于 `window_size`。

测试集不会参与标准化拟合、模型训练或阈值计算。

### 3.3 `decision_function(X_test) -> numpy.ndarray`

返回每个时间点的连续异常分数。

- 输入：形状 `(n, 38)` 的测试数据；
- 返回：形状 `(n,)`、类型为 `float32` 的分数数组；
- 分数越大表示越异常；
- 前 `window_size` 个位置因历史数据不足，分数为 `0`；
- 后续分数是标准化空间中预测值与真实下一点之间的 38 维均方误差。

调用前必须先调用 `fit`，否则抛出 `RuntimeError`。

### 3.4 `predict(X_test) -> numpy.ndarray`

根据固定训练阈值生成二值异常标签。

- 输入：形状 `(n, 38)` 的测试数据；
- 返回：形状 `(n,)` 的整数数组；
- `0`：正常；
- `1`：异常；
- 当分数严格大于 `threshold_` 时判定为异常。

调用前必须先调用 `fit`。

### 3.5 `reset_stream() -> None`

清空流式检测缓存。

开始处理一段新的独立数据流前必须调用此函数，避免上一段数据的历史窗口影响
当前数据流。

### 3.6 `step_stream(point) -> int`

接收一个新的 38 维观测点并立即返回当前点的检测结果。

- 输入：形状 `(38,)` 的单个观测点；
- 返回：整数 `0` 或 `1`；
- 前 `window_size` 个点只用于积累历史，返回 `0`；
- 从第 `window_size + 1` 个点开始，使用前 `window_size` 个真实点预测当前点；
- 当前点判断完成后才加入历史缓存；
- 缓存只保存真实观测值，不保存预测值。

示例：

```python
import numpy as np
from models.kan_ad_detector import KANADDetector

train = np.loadtxt("data/train/machine-1-1.txt", delimiter=",").astype(np.float32)
test = np.loadtxt("data/test/machine-1-1.txt", delimiter=",").astype(np.float32)

detector = KANADDetector(window_size=10, n_harmonics=5)
detector.fit(train)

scores = detector.decision_function(test)
labels = detector.predict(test)

detector.reset_stream()
stream_labels = [detector.step_stream(point) for point in test]
```

## 4. 测试程序

### 4.1 `tests/test_kan_ad_detector.py`

这是轻量级单元测试，不读取完整 SMD 数据，也不进行耗时的真实模型训练。

测试内容包括：

- 离线 `decision_function` 输出形状和数据类型；
- 前 `window_size` 个位置分数为 `0`；
- 离线检测和流式检测的时间对齐；
- 异常点能产生异常标签；
- 未训练时调用推理会抛出 `RuntimeError`；
- 输入维度错误时会抛出 `ValueError`。

测试内部使用轻量假模型，目的是隔离检查 detector 的接口、缓存和时间对齐逻辑。

### 4.2 `tests/validate_smd.py`

这是使用真实 SMD 数据的端到端验证程序，读取：

- `data/train/machine-1-1.txt`；
- `data/test/machine-1-1.txt`；
- `data/test_label/machine-1-1.txt`。

验证内容包括：

- 数据形状是否为训练集 `(n, 38)`、测试集 `(n, 38)`；
- 数据是否包含非有限值；
- 使用训练集训练 KAN-AD；
- 测试集离线分数和标签生成；
- 全量逐点流式结果是否与离线结果一致；
- Precision、Recall、F1 和 AUROC。

默认使用 `1` 个 epoch，适合快速验证流程。正式实验可以增加训练轮数。

## 5. 如何启动

### 5.1 激活虚拟环境

PowerShell：

```powershell
.\venv\Scripts\Activate.ps1
```

也可以直接使用虚拟环境中的 Python，不需要激活：

```powershell
.\venv\Scripts\python.exe --version
```

### 5.2 运行轻量单元测试

```powershell
.\venv\Scripts\python.exe .\tests\test_kan_ad_detector.py
```

或者使用 unittest 自动发现：

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 5.3 运行真实 SMD 验证

```powershell
.\venv\Scripts\python.exe .\tests\validate_smd.py
```

指定训练轮数和批大小：

```powershell
.\venv\Scripts\python.exe .\tests\validate_smd.py `
    --epochs 10 `
    --batch-size 8192
```

如果只需要验证离线推理，可以跳过逐点流式检查：

```powershell
.\venv\Scripts\python.exe .\tests\validate_smd.py --skip-stream
```

## 6. 依赖

项目虚拟环境中需要安装：

```powershell
python -m pip install numpy scikit-learn torch torchinfo tqdm easytsad
```

# KDA-MLA Stock

基于 PyTorch 的股票收益预测研究项目，使用 Kimi Delta Attention（KDA）与 Multi-head
Latent Attention（MLA）编码单只股票的历史 OHLCV 特征，并预测未来 5 个交易日收益。

项目默认使用真实 Qlib `CSI300` 时变成分股数据。合成数据仅用于检查代码能否运行，不应拿来判断策略有效性。

> 本项目用于模型和量化研究，不构成投资建议。测试集结果也不等于实盘收益。

## 模型概览

默认 `configs/model-small.json` 包含 12 层、384 隐藏维度的约 2218 万参数模型：

```text
OHLCV
  -> 10 个无未来信息的时序特征
  -> 线性特征投影
  -> 9 层 KDA + 3 层 MLA
  -> 最后一个交易日的时序状态
  -> 未来 5 日收益
  -> Rank IC / ICIR / 多空回测
```

KDA 负责用递归状态压缩长序列，MLA 在少量层中补充完整的因果注意力交互。该项目借用了 KDA 和 MLA
模块，不是 Kimi K3 的缩小版，也没有使用语言模型、Tokenizer 或文本预训练。

## AutoDL 快速开始

推荐 Python 3.10 或 3.11、Linux、NVIDIA CUDA 环境。进入项目目录后安装：

```bash
cd /root/kda-mla-stock
pip install -U pip
pip install -e ".[qlib,cuda]"
```

`cuda` 额外依赖会安装 `fla-core` 的 KDA CUDA/Triton 内核。纯 PyTorch 后端可以在 CPU 上运行测试，
但不适合训练长度为 256 的默认模型。

下载真实 Qlib A 股数据、导出 `CSI300` 并直接开始训练：

```bash
bash scripts/train_real.sh
```

脚本执行两个明确步骤：

```bash
python scripts/prepare_qlib_real.py
python scripts/train.py --train-config data/train-real.json
```

数据准备脚本不会默认覆盖已有 Qlib 数据。需要重新下载时显式执行：

```bash
python scripts/prepare_qlib_real.py --force-download
```

它根据下载到的真实交易日自动划分数据：最后 252 个交易日作为测试集，之前 252 个交易日作为验证集，
更早的数据用于训练。生成的实际配置保存在 `data/train-real.json`。

## 查看损失曲线

训练终端使用 `tqdm` 按 batch 显示进度、当前 loss、平均 loss 和学习率，同时把 loss、验证集
Rank IC 和学习率写入 TensorBoard：

```bash
tensorboard --logdir outputs/kda-mla-small/tensorboard --host 0.0.0.0 --port 6006
```

在 AutoDL 控制台映射 6006 端口后即可查看。每轮的原始数值也保存在
`outputs/kda-mla-small/history.json`。

## 断点续训

训练每轮保存 `last.safetensors` 和优化器状态，验证损失最优的权重保存为 `best.safetensors`：

```bash
python scripts/train.py \
  --train-config data/train-real.json \
  --resume outputs/kda-mla-small
```

## 测试集评估

训练结束后运行严格的时间外测试：

```bash
python scripts/evaluate.py \
  --checkpoint-dir outputs/kda-mla-small \
  --split test
```

输出目录为 `outputs/kda-mla-small/evaluation_test/`，包括：

- `predictions.csv`：逐日逐股票预测和真实收益；
- `daily_ic.csv`：每天的 Pearson IC 和 Rank IC；
- `backtest.csv`：多空收益、换手率、成本、净值和回撤；
- `summary.json`：MSE、MAE、方向准确率、Rank IC、ICIR、Sharpe 和最大回撤。

回测每天按预测值选择前后 20% 的股票，默认收取 10 bps 换手成本。由于标签是未来 5 日收益，
回测每 5 个交易日再平衡一次，避免把重叠标签当作独立的每日收益。

## 数据设计

标准 CSV 字段如下：

```text
date,symbol,open,high,low,close,volume
```

价格应使用一致的复权口径。每个 `date + symbol` 只能有一行。项目生成以下 10 个特征：

- 1、5、20 日收益；
- 日内开收收益和最高最低振幅；
- 1 日 `log1p` 成交量变化和 20 日成交量 Z-score；
- 20 日收益波动率；
- 收盘价相对 MA5、MA20 的偏离。

所有滚动特征只使用当前及历史数据。归一化均值和标准差只在训练期拟合。训练样本的标签结束日期必须
不晚于训练截止日，验证样本同理，从而清除跨边界标签泄漏。

如果已经有 Qlib 数据，只导出而不下载：

```bash
python scripts/export_qlib.py \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --market csi300 \
  --output data/qlib-csi300.csv
```

美股实验可安装并使用 Yahoo Finance：

```bash
pip install -e ".[data]"
python scripts/download_yfinance.py AAPL MSFT NVDA AMZN META GOOGL \
  --start 2010-01-01 \
  --output data/us-tech.csv
```

手工指定一组今天仍存在的股票会产生幸存者偏差，因此正式实验优先使用带历史成分区间的 Qlib 市场。

## 配置与消融

模型配置：

- `configs/model-small.json`：默认 9 KDA + 3 MLA 混合模型；
- `configs/model-kda-only.json`：纯 KDA 对照；
- `configs/model-mla-only.json`：纯 MLA 对照；
- `configs/model-smoke.json`：CPU 冒烟测试小模型。

使用相同数据切分比较三种结构：

```bash
python scripts/train.py \
  --model-config configs/model-kda-only.json \
  --train-config data/train-real.json \
  --output-dir outputs/kda-only
```

```bash
python scripts/train.py \
  --model-config configs/model-mla-only.json \
  --train-config data/train-real.json \
  --output-dir outputs/mla-only
```

不要只比较训练损失。主要比较相同测试期的 Rank IC、ICIR、含成本 Sharpe、最大回撤和换手率。

## 显存与速度

默认模型约 2218 万参数，30 GB 显存足够进行单卡训练。默认批次为 64、序列长度为 256；实际显存还取决于
PyTorch、CUDA、Triton 和 GPU 型号。发生 OOM 时先把 `batch_size` 改为 32 或 16。

确认日志没有提示 KDA 落入慢速 PyTorch 路径。也可以把模型配置中的 `attention_backend` 设置为
`"fla"`，让缺少 `fla-core` 时直接报错，而不是静默使用参考实现。

## 代码验证

安装开发依赖并运行检查：

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

仅用于检查完整训练链路的合成数据命令：

```bash
python scripts/generate_synthetic.py \
  --start 2020-01-01 \
  --days 1000 \
  --symbols 8
python scripts/train.py \
  --model-config configs/model-smoke.json \
  --train-config configs/train-smoke.json
```

## 项目结构

```text
kda-mla-stock/
├── configs/                  # 混合模型、消融和训练配置
├── scripts/                  # 真实数据准备、训练、评估与辅助命令
├── src/kda_mla_stock/
│   ├── data.py               # 特征、归一化、时间切分和窗口数据集
│   ├── modeling.py           # KDA、MLA 与收益预测模型
│   ├── training.py           # 训练、TensorBoard 和 safetensors 检查点
│   ├── metrics.py            # IC、Rank IC 和 ICIR
│   └── backtest.py           # 含换手成本的多空回测
└── tests/                    # 泄漏、模型、指标、回测和训练测试
```

真实研究还需要处理涨跌停、停牌成交可行性、手续费/滑点、指数成分变更、退市样本和数据发布时点。
当前回测是模型研究基线，不是完整撮合引擎。

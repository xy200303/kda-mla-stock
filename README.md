# KDA-MLA Stock

股票收益预测研究项目，使用 Kimi Delta Attention（KDA）与 Multi-head Latent Attention（MLA）
编码单只股票的历史 OHLCV 特征，并预测未来 5 个交易日收益。项目同时提供 sklearn 和 LightGBM
传统基线，以相同数据切分、Qlib 回测和统计检验进行公平比较。

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

模型算法不重复造轮子：Ridge、Random Forest 和 HistGradientBoosting 直接使用 `scikit-learn`，
LightGBM 使用官方 `lightgbm`，KDA 正式训练使用 `fla-core`，MLA 使用 PyTorch 官方 SDPA。项目自己的
代码只负责模型组合、无泄漏特征、统一训练评估和科研报告。

所有模型实现位于 `src/kda_mla_stock/models/`，每种架构拥有独立目录。`MarketDataModule` 统一提供
时序和表格数据视图，`TrainRunner`、`EvaluationRunner` 负责组装 Trainer、Validator 和最终评估器。
Ridge、Random Forest、HistGBDT 与 LightGBM 仍直接使用官方库实现。

## AutoDL 快速开始

推荐 Python 3.10 或 3.11、Linux、NVIDIA CUDA 环境。进入项目目录后安装：

```bash
cd /root/kda-mla-stock
pip install -U pip
pip install -e ".[qlib,cuda,ml]"
```

`cuda` 额外依赖会安装 `fla-core` 的 KDA CUDA/Triton 内核。纯 PyTorch 后端可以在 CPU 上运行测试，
但不适合训练长度为 256 的默认模型。`ml` 会安装 sklearn、LightGBM 和 joblib。

下载真实 Qlib A 股数据、导出 `CSI300` 并训练约 712 万参数的快速模型：

```bash
bash scripts/train_fast.sh
```

快速脚本使用批次 128、4 个数据加载进程和强制 FLA 内核。需要训练原始 2218 万参数完整模型时运行：

```bash
bash scripts/train_real.sh
```

数据准备脚本不会默认覆盖已有 Qlib 数据。需要重新下载时显式执行：

```bash
python scripts/prepare_qlib_real.py --force-download
```

下载使用固定的 `.part` 临时文件，并在网络中断后自动断点续传（默认最多 8 次）。如果所有重试都失败，
直接重新执行同一命令即可从已下载位置继续；可通过 `--download-retries` 和 `--download-timeout`
调整重试次数和读取超时。

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

训练每轮保存 `last.safetensors` 和优化器状态，验证 Rank IC 最优的权重保存为 `best.safetensors`：

```bash
python scripts/train.py \
  --model-config outputs/kda-mla-small/model_config.json \
  --train-config data/train-real.json \
  --output-dir outputs/kda-mla-small \
  --batch-size 128 \
  --num-workers 4 \
  --train-stride 5 \
  --epochs 30 \
  --patience 6 \
  --resume outputs/kda-mla-small
```

这里从输出目录读取旧模型配置，因此能继续原 2218 万参数检查点。断点位于 epoch 边界；中途停止某个
epoch 时，该 epoch 已完成的 batch 不会恢复。

## 模块性能分析

在 AutoDL CUDA 环境中使用正式训练的 batch、序列长度和精度，分别测量 KDA、MLA、FFN、两类
编码层以及完整模型的前向和反向耗时：

```bash
python scripts/profile_model.py \
  --model-config configs/model-fast.json \
  --train-config data/train-real.json \
  --output-dir outputs/profile-fast
```

终端会输出各模块的参数量、前向耗时、前向加反向耗时、吞吐量和峰值显存。
`module_timings.json` 保存模块基准结果，`operator_table.txt` 列出耗时最高的 PyTorch/CUDA 算子，
`trace.json` 可以使用 Chrome/Perfetto trace viewer 查看时间线。首次 FLA/Triton 编译发生在预热阶段，
不计入正式计时。

高性能版本将 KDA 的 Q/K/V/G 投影和深度卷积各合并为一次算子调用，将 FFN 的 gate/up 投影合并，
缓存 MLA RoPE，并减少训练循环中的 GPU/CPU 同步。参数量和数学结构保持不变，旧版 safetensors
权重会在加载时自动合并到新布局。优化前后速度应使用该 profile 在同一块 GPU、相同 batch 下比较。

## 测试集评估

训练结束后运行严格的时间外测试：

```bash
python scripts/evaluate.py \
  --checkpoint-dir outputs/kda-mla-small \
  --split test \
  --qlib-provider-uri ~/.qlib/qlib_data/cn_data
```

输出目录为 `outputs/kda-mla-small/evaluation_test/`，包括：

- `predictions.csv`：逐日逐股票预测和真实收益；
- `daily_ic.csv`：每天的 Pearson IC 和 Rank IC；
- `qlib_portfolio_report.csv`：Qlib 每日收益、基准、成本和换手；
- `qlib_risk_analysis.csv`：策略、基准和含成本超额收益风险指标；
- `qlib_trade_indicators.csv`：成交率、价格优势等执行指标；
- `training_curves.png`：损失、Rank IC 和训练吞吐；
- `prediction_diagnostics.png`：IC、累计 Rank IC、预测散点和十分位收益；
- `qlib_portfolio_performance.png`：净值、超额收益、回撤和换手；
- `summary.json`：预测指标、诊断回测和 Qlib 正式回测的统一摘要。

Qlib 正式回测使用 `TopkDropoutStrategy + SimulatorExecutor + Exchange`：默认持有前 50 只股票，
每日最多替换 10 只并至少持有 5 日。T 日信号在 T+1 日开盘执行，同时模拟停牌、涨跌停、买卖费用
和最低手续费，并与沪深 300 基准比较。项目原有的简化多空结果仍保存为 `backtest.csv`，只作为诊断，
不作为论文主要策略结论。

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

- `configs/model-fast.json`：约 712 万参数的 6 KDA + 2 MLA 推荐快速模型；
- `configs/model-small.json`：约 2218 万参数的 9 KDA + 3 MLA 完整模型；
- `configs/model-kda-only.json`：纯 KDA 对照；
- `configs/model-mla-only.json`：纯 MLA 对照；
- `configs/model-lstm.json`、`model-gru.json`：循环网络基线；
- `configs/model-transformer.json`：标准因果 Transformer 基线；
- `configs/model-mlp.json`：低成本 MLP 基线；
- `configs/model-smoke.json`：CPU 冒烟测试小模型。
- `configs/ml-ridge.json`：sklearn Ridge 线性基线；
- `configs/ml-random-forest.json`：sklearn Random Forest 基线；
- `configs/ml-hist-gbdt.json`：sklearn HistGradientBoosting 基线；
- `configs/ml-lightgbm.json`：官方 LightGBM 基线。

传统模型使用相同的 256 日窗口。每个原始特征提取当前值，以及 5、20、60、256 日的均值和标准差，
共 `10 + 4 x 2 x 10 = 90` 维；所有统计量都只观察锚点当日及之前。训练和评估仍使用统一命令，模型
类型由配置自动识别：

```bash
python scripts/train.py \
  --model-config configs/ml-lightgbm.json \
  --train-config data/train-real.json \
  --output-dir outputs/lightgbm
python scripts/evaluate.py \
  --checkpoint-dir outputs/lightgbm \
  --split test
```

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

批量运行单种子基线和消融：

```bash
python scripts/run_experiments.py \
  --experiments kda-mla-fast kda-only mla-only lstm gru transformer mlp \
                ridge random-forest hist-gbdt lightgbm \
  --seeds 42
```

正式论文实验至少使用三个随机种子：

```bash
python scripts/run_experiments.py \
  --experiments kda-mla-fast kda-mla-full kda-only mla-only transformer lstm \
                ridge random-forest hist-gbdt lightgbm \
  --seeds 42 3407 2026
```

汇总目录 `outputs/paper/` 会生成逐次结果、均值/标准差表、模型对比图和日度 Rank IC 配对块自助法
置信区间。完整实验口径见 [科研实验方案](docs/RESEARCH_PROTOCOL.md)。

## 显存与速度

原完整模型约 2218 万参数。你当前服务器日志为 6.62 batch/s、批次 64，即约 424 samples/s，
每轮约 28.6 分钟。由于标签是未来 5 日收益，新配置让训练集按 5 日步长抽取锚点，避免相邻标签
高度重叠；验证和测试仍逐日评估。仅这一项就会把每轮训练样本从约 72.8 万降到约 14.6 万。

同时启用批次 128、4 个加载进程、fused AdamW、BF16 和 TF32；实际提升取决于 GPU，训练日志会输出
`throughput` 和 `peak_memory`，应以服务器实测为准。30 GB 显存发生 OOM 时先回退批次 64；7.1M
快速模型通常还能继续增大批次。需要复现全量重叠窗口时使用 `--train-stride 1`。

正式 KDA 配置已设置 `attention_backend="fla"`，缺少 `fla-core` 会直接报错。启动日志必须显示
`KDA execution backend: fla-core`。`--compile-mode reduce-overhead` 可在单独速度实验中尝试，但
`torch.compile` 与 Triton/FLA 组合依赖具体版本，因此没有默认开启。

## 代码验证

安装开发依赖并运行检查：

```bash
pip install -e ".[dev,ml]"
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
│   ├── core/                 # 配置、运行时、接口契约与制品读写
│   ├── data/
│   │   ├── market.py         # CSV 校验、特征工程与归一化
│   │   ├── window.py         # 时间切分与时序窗口数据集
│   │   ├── tabular.py        # 传统模型的因果聚合特征
│   │   └── module.py         # 统一 MarketDataModule
│   ├── models/
│   │   ├── registry.py       # 架构、后端与模型制品注册表
│   │   ├── kda_mla/          # KDA、MLA 和混合预测模型
│   │   ├── lstm|gru|.../     # 每个神经基线的独立实现
│   │   └── ridge|.../        # sklearn/LightGBM 官方模型构造器
│   ├── training/             # Torch 与 estimator Trainer
│   ├── validation/           # 训练期 Torch 与 estimator Validator
│   ├── evaluation/           # 最终指标、Qlib 回测与科研图表
│   └── orchestration/        # 统一训练、评估与实验调度
└── tests/                    # 泄漏、模型、指标、回测和训练测试
```

新增模型时，在 `models/<architecture>/` 中实现模型并在 `models/registry.py` 注册。调度层根据注册项
选择 Torch 或 estimator Trainer，不需要修改数据切分、Validator 或最终评估流程。

真实研究还需要处理涨跌停、停牌成交可行性、手续费/滑点、指数成分变更、退市样本和数据发布时点。
当前回测是模型研究基线，不是完整撮合引擎。

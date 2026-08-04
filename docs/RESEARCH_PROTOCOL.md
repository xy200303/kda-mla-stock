# 科研实验方案

本文档固定数据、训练、消融和回测口径，用于生成可复现的论文实验结果。任何偏离此方案的运行都应记录在实验日志中。

## 研究问题

1. KDA 与 MLA 混合编码器是否比传统机器学习、LSTM、GRU、Transformer 和 MLP 基线具有更高的截面预测能力？
2. KDA-only 与 MLA-only 相对混合结构的性能变化，能否说明两类注意力模块存在互补作用？
3. 约 712 万参数的快速模型能否以较小精度损失换取显著训练吞吐提升？
4. 模型预测优势在交易费用、涨跌停、停牌和换手约束下是否仍然存在？

## 数据与防泄漏

- 数据源：Qlib 中国市场日频数据，股票池为时变 `CSI300` 成分股。
- 输入：每只股票过去 256 个交易日的 10 个 OHLCV 派生特征。
- 标签：未来 5 个交易日收益率。
- 划分：最后 252 个交易日为测试集，之前 252 个交易日为验证集，更早数据为训练集。
- 归一化：均值与标准差只使用训练期有效行拟合。
- 清除边界：标签结束日期不得越过训练或验证截止日。
- 测试集只用于最终报告，不参与模型选择或超参数调整。

当前服务器导出的数据快照包含 727,641 个候选训练窗口、63,751 个验证窗口和 66,176 个测试窗口。正式配置按 5 日步长抽取训练锚点，预计实际训练窗口约 14.6 万个；验证和测试仍保留逐日窗口。重新下载数据后，需在论文中记录实际日期范围、Qlib 数据版本、候选样本数和抽样后的样本数。

## 模型组

| 组别 | 配置 | 参数量 | 用途 |
| --- | --- | ---: | --- |
| KDA-MLA Fast | `model-fast.json` | 7,123,993 | 推荐主模型与速度实验 |
| KDA-MLA Full | `model-small.json` | 22,180,279 | 容量上限对照 |
| KDA-only | `model-kda-only.json` | 23,215,945 | 移除 MLA 的消融 |
| MLA-only | `model-mla-only.json` | 19,073,281 | 移除 KDA 的消融 |
| LSTM | `model-lstm.json` | 834,305 | 循环网络基线 |
| GRU | `model-gru.json` | 634,113 | 轻量循环网络基线 |
| Transformer | `model-transformer.json` | 3,162,625 | 标准因果注意力基线 |
| MLP | `model-mlp.json` | 139,777 | 低成本非序列基线 |
| Ridge | `ml-ridge.json` | 拟合后记录 | sklearn 线性基线 |
| Random Forest | `ml-random-forest.json` | 拟合后记录节点数 | sklearn Bagging 树基线 |
| HistGBDT | `ml-hist-gbdt.json` | 拟合后记录节点数 | sklearn Boosting 树基线 |
| LightGBM | `ml-lightgbm.json` | 拟合后记录叶子数 | 官方 LightGBM 基线 |

神经模型参数量与树模型节点数不是同一种复杂度指标，不应直接解释为等参数比较。论文必须分别报告神经
模型参数量、传统模型结构复杂度、拟合时间、推理时间、训练吞吐和峰值显存。若需要严格的等参数比较，
应另建 parameter-matched 配置，不应只通过模型名称宣称公平。

传统模型算法直接来自 `scikit-learn` 和官方 `lightgbm`，不使用项目自写的回归器或决策树。输入由同一
`MarketDatasetBundle` 生成：10 个标准化特征的当前值，加上 5、20、60、256 日均值和标准差，共 90
维。滚动窗口只能读取预测锚点及之前的数据，训练、验证、测试引用与神经模型完全一致。

## 训练协议

- 优化器：AdamW；CUDA 可用时启用 fused 实现。
- 精度：BF16；允许 TF32 矩阵乘。
- 批次：30 GB 显存先使用 128，OOM 时回退到 64。
- 训练步长：每只股票每 5 日取一个锚点，使相邻未来五日标签不重叠；所有模型使用相同步长。
- 最大轮数：30；验证 Rank IC 连续 6 轮不提升时早停。
- 模型选择：验证集 Rank IC 最大的检查点。
- 随机种子：正式表格至少使用 `42 3407 2026` 三个种子，报告均值和标准差。
- KDA 正式实验强制 `fla-core`，不得使用逐时间步的 PyTorch 参考路径提交速度结果。
- 传统模型使用配置中固定的 sklearn/LightGBM 超参数；LightGBM 只在验证集早停，不读取测试集。
- 神经与传统模型都通过统一 `Trainer`、`Valer` 和数据加载器运行，不允许为某个基线单独改变切分或特征归一化。

快速完成第一轮筛选：

```bash
python scripts/run_experiments.py \
  --experiments kda-mla-fast kda-only mla-only lstm gru transformer mlp \
                ridge random-forest hist-gbdt lightgbm \
  --seeds 42
```

确定候选模型后运行三种子实验：

```bash
python scripts/run_experiments.py \
  --experiments kda-mla-fast kda-mla-full kda-only mla-only transformer lstm \
                ridge random-forest hist-gbdt lightgbm \
  --seeds 42 3407 2026
```

## Qlib 回测协议

- 策略：`TopkDropoutStrategy`，持有预测排名前 50 的股票。
- 换仓：默认每日最多替换 10 只，最低持有 5 个交易日，与五日预测周期对齐。
- 信号时序：T 日收盘后形成信号，Qlib 在 T+1 日读取该信号并按开盘价成交。
- 交易约束：只交易可成交股票，涨跌停阈值 9.5%。
- 成本：买入 5 bps、卖出 15 bps、最低手续费 5 元。
- 资金与基准：初始资金 1 亿元，基准为 `SH000300`。

主要预测指标为 MSE、MAE、方向准确率、IC、Rank IC 和年化 Rank ICIR。主要投资指标为含成本年化收益、相对基准的信息比率、最大回撤、换手率、成交率和价格优势。

## 统计与图表

每个实验会生成训练损失、验证 Rank IC、吞吐、日度 IC、累计 Rank IC、预测分位数组合收益、策略净值、基准净值、超额收益、回撤和换手图。`compare_experiments.py` 还会输出：

- `comparison_runs.csv`：每个随机种子的原始结果；
- `comparison.csv` 与 `comparison.md`：模型均值和标准差；
- `model_comparison.png`：核心指标横向图；
- `rank_ic_significance.csv`：相对主模型的配对块自助法 95% 置信区间；
- `rank_ic_significance.png`：Rank IC 差异森林图。

块自助法默认按 20 个交易日分块、重复 2,000 次，以减少日度指标自相关导致的置信区间偏窄。统计显著不等于经济显著，结论必须同时结合含成本收益与回撤。

## 论文中必须披露的限制

- Qlib 离线数据不是实时行情，结果取决于下载快照日期。
- 当前特征未显式建模财报发布时间、行业中性化和风险暴露。
- 单一 CSI300 股票池不能证明模型可泛化到中小盘或其他市场。
- 回测不能完全还原实际冲击成本、排队成交和实盘延迟。
- 多次尝试超参数会造成选择偏差，应记录全部实验而不是只保留最好结果。

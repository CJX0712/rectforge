# RectForge Model Card

## 模型

- **名称**: RectFuse（旗舰）/ cfm_vp / ddpm / ddim（基线）
- **架构**: TimeCondMLP —— 正弦时间嵌入（dim 32）→ SiLU MLP（hidden 128 × 3 层，完整配置）→ 线性输出
- **参数量**: ~50K（完整配置），纯 NumPy 前向 + 解析反向
- **范式**: 流匹配（conditional flow matching）—— 线性概率路径，minibatch-OT 耦合，
  uniform 时间采样；采样走 RK4 概率流 ODE（γ=0，精确）

## 训练

- 目标：`||v_θ(x_t,t) − (x1−x0)||²`（x_t=(1−t)x0+t·x1，OT 耦合后配对）
- 优化：Adam，lr 2e-3，warmup 100 + cosine 衰减至 5%，梯度范数裁剪 1.0
- 完整配置：2400 步 × batch 256，数据 8192 样本/DGP/seed
- 确定性：BLAS 钉 1 + 显式 Generator + 固定 init 种子；两次运行逐位相同（G3）

## 意图用途

- 生成建模教学与算法研究：每条概率路径、每个采样器、每个评测口径都可独立审计
- 算法对照实验的确定性基建设施：等 NFE 配对比较、门禁自动化
- 低资源环境（CPU-only、无 GPU/框架依赖）下的可复现实验

## 非意图用途

- 图像/文本等高维真实数据（MLP + 2D 合成 DGP 仅为算法验证尺度）
- 生产部署

## 评测

- KL(p_true‖q̂)：q̂ 为模型样本 Silverman KDE；**注意协议地板**（KL 估计有样本量相关正下界，
  gmm8@4096 ≈ 0.55），绝对值须对照地板解读
- 无偏 MMD²（中位数带宽）、sliced W1（黄金角投影）
- 门禁：G1a 归一化 / G1b KL 非负 / G2 等 NFE 配对 / G2b NFE 效率 / G3 逐位确定性

## 已知局限

- SI 路径（γ>0）的经验混合分数修正随语料规模 O(n) 增长
- checker 密度盒界不连续，归一化自检容差放宽至 2e-3（求积性质，如实记录）
- DDIM 在低 NFE 下对 t≈1 端的 x0 估计误差敏感（1/√ab_T 放大），基线对比时按等 NFE 口径解读

## 训练数据

全部为合成 2D 分布（gmm8/ring/wave/checker），无任何人工数据，无隐私风险。

## 引用

```bibtex
@software{rectforge2026,
  author = {晨星 (Chenxing)},
  title = {RectForge: deterministic NumPy flow matching foundry},
  year = {2026},
  url = {https://github.com/CJX0712/rectforge}
}
```

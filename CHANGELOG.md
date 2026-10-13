# Changelog

本项目的所有显著变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.1.0] - 2026-10-13

### Added
- 三条概率路径：LinearPath（rectified flow）/ VPPath（cosine CFM）/ SIPath（随机插值）
- minibatch-OT 耦合（scipy `linear_sum_assignment` 精确解，不可用自动降级）
- TimeCondMLP：正弦时间嵌入 + SiLU MLP，纯 NumPy 解析反向传播（梯度检验 <1e-6 硬门禁）
- DDPM ancestral / DDIM 基线（cosine schedule，β 截断重建防除零）
- Euler/RK4 ODE 积分器 + NFE 记账 + SI 路径精确 PF-ODE 修正
- 评测：KL(KDE Silverman) / 无偏 MMD² / sliced W1（黄金角投影）
- 5 个解析密度（gmm8/ring/wave/checker/gauss2gauss），归一化自检 + 采样同源铁律
- 基准流水线：等 NFE 配对比较 + 门禁 G1a/G1b/G2/G2b/G3/G4 + 消融（含负结果）
- 60+ 单测：梯度硬门禁、解析输运金标准、积分器收敛阶、逐位确定性
- CLI：run / densities / selftest / version；ENV_RECTFORGE_* 全量配置覆盖
- CI（GitHub Actions）：ruff + pytest + demo 门禁
- Dockerfile、Makefile、完整文档（architecture / model card）

### Fixed
- checker 采样器与密度同源（拒绝采样替换纯 uniform——FlowForge 铁律）
- DDIM 时间网格方向反转导致的样本发散（1e14 量级）
- cosine schedule ab[T]=0 精确除零（从截断 β 重建 ab）
- SI 修正项 t=0 精确评估的 0/0（s² 下限 1e-12）

### 已知取舍（如实记录）
- checker 盒界不连续 → 归一化梯形误差 O(spacing)≈3e-4，容差 2e-3
- logit-normal 时间采样在小规模训练下饿死两端 t（实测 ratio 1.67），旗舰改用 uniform
- KL 评测存在样本量相关协议地板（gmm8@4096≈0.55），门禁全部采用配对相对口径

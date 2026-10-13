# RectForge 架构

## 总体数据流

```
DGP(densities) ──► 训练(trainer) ──► 采样(sampling/baselines) ──► 评测(eval) ──► 门禁(pipeline)
     │                  │                    │                        │               │
  解析 log p     TimeCondMLP+Adam        ODE/ancestral           KL/MMD²/W1     G1a-G4 硬阈值
  归一化自检     解析反向传播            NFE 记账                KDE Silverman  benchmark.json
```

## 模块边界与不变量

### core/
- `seed.py`：`pin_threads()` 六个 BLAS env 钉 1（必须在 numpy 导入前——`__init__` 已尽力；
  严格确定性由 conftest/demo 顶部再次保证）；`set_all(seed)` 返回显式 Generator。
- `errors.py`：错误目录 E000-E599。注册时同时检测**重码与重名**（ValuaForge 教训：
  同名类会静默遮蔽先定义者）。`ConfigError` 属 E103 族。
- `types.py`：`TrainConfig` / `BenchmarkConfig` / `EvalPoint` / `GateReport`。
- `config.py`：`ENV_RECTFORGE_*` 覆盖。按字段名做类型映射（PEP 563 下 `f.type` 是字符串，
  MatrixForge 教训），覆盖后 schema 校验（E104-E111），`train_velocity` 拒绝 si+ot 组合（E112——
  SI 路径的精确 PF-ODE 修正要求 independent 耦合）。

### data/densities.py —— FlowForge 铁律
每个合成密度必须：① 解析 `log_prob`；② 数值归一化自检（2D 复合梯形，谱精度）；
③ **采样器与 log_prob 严格同源**。checker 曾违反 ③（纯 uniform 采样 vs 调制密度），
KL(p‖p) 结构性为负——已改拒绝采样并加协议地板测试。

| 密度 | 用途 | 归一化容差 |
|---|---|---|
| gmm8（8 模态环） | 多模态主战场 | 1e-4 |
| ring | 细圆环（1/r 极坐标因子） | 1e-4 |
| wave | 条件正弦流形 | 1e-4 |
| checker | 盒内调制密度（盒界不连续 → 梯形 O(spacing)，实测 3e-4，如实记录） | 2e-3 |
| gauss2gauss | 金标准：闭式速度场 `gaussian_transfer_velocity` | 1e-4 |

### models/mlp.py
`TimeCondMLP`：正弦时间嵌入 + SiLU MLP + 输出层小初始化（训练初期 v≈0）。
解析反向传播，**梯度检验硬门禁：中心差分最大绝对误差 < 1e-6**（实测 ~3e-10，
FD 噪声口径；相对误差口径在小梯度分量上会虚高 1e-2——诊断教训）。

### paths/
约定：`t=0` 数据端、`t=1` 噪声端；ODE 积分 `t:1→0`（生成方向）。
- LinearPath：`x_t=(1-t)x0+t·x1`，v=x1-x0。γ=0 ⇒ 纯概率流 ODE，采样精确。
- VPPath：cosine 路径 CFM，v = da·x0+ds·x1（有限差分验证）。
- SIPath：+γ(t)ε，γ=σt(1-t)。γ>0 时 PF-ODE 需 `-½g²∇log p_t` 修正（sampling/ode.py 闭式，
  经验混合近似）；**仅允许 independent 耦合**（E112）。

### paths/coupling.py
`ot_pair`：scipy `linear_sum_assignment` 精确解 minibatch OT（Tong 2023）。
scipy 不可用自动降级 independent。测试断言 OT 成本 ≤ 随机配对。

### training/trainer.py
dict 版 Adam（偏差修正首步步长=lr 有专测）、warmup+cosine 调度、全局范数裁剪。
训练随机流：`rng.integers` 有放回取子批（避免 epoch 边界效应）。

### baselines/ddpm.py
cosine schedule（Nichol & Dhariwal 2021）。**关键实现细节**：cosine 的 ab[T]=cos(π/2)²=0
精确为零 → x0 估计除零发散（DDIM 样本爆到 1e14 的根因）。标准做法从截断 β∈[1e-4,0.999]
重建 ab。DDIM 时间网格必须从噪声端 t=T-1 走向数据端 t=0（方向反了同样爆炸）。

### sampling/ode.py
Euler/RK4（t:1→0）+ NFE 记账（RK4 每步 4 次）。SI 修正项 `0.5(g²/s²)Σw_i(m_i-x)`，
`s²=t²+g²` 下限 1e-12 防 RK4 中间级恰取 t=0 时的 0/0。

### eval/metrics.py
- `kl_p_q`：KL(p_true‖q̂)，q̂=模型样本 Silverman KDE，common random numbers 配对。
  **协议地板**：KL 估计本身有样本量相关的正下界（gmm8@4096 ≈ 0.55、checker@2048 ≈ 0.03），
  绝对值解读必须对照地板；G2 门禁只用配对相对比较。
- `mmd2`：无偏估计 + 中位数带宽。
- `sliced_w1`：黄金角确定性投影（诚实命名——scipy `wasserstein_distance` 是 W1）。

### pipeline/benchmark.py
配对设计：同 (DGP,seed) 下三模型共用训练随机流；模型 init 种子固定；
true 样本与初始噪声全方法共享。门禁 G1a/G1b/G2/G2b 见 README；G3 确定性二次全流程逐位比较；
G4 消融 5 变体（full / no_ot / lognormal_time / euler / si_0.25），**负结果如实报告**。

## 确定性铁律清单

1. BLAS 线程钉 1（六个 env，numpy 导入前）
2. 显式 Generator，无全局随机
3. init 种子固定（_MODEL_INIT_SEED=1000），训练流按 seed 变
4. 评测 common random numbers
5. G3：两次全流程 max|Δ| == 0.0（非容差，逐位）

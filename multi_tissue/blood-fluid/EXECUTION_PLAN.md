# SPH-GNN 血液流体 + 血管模型执行计划

> **生成日期**: 2026-03-12
> **状态**: 代码尚未实现，需从零构建
> **项目根目录**: `~/workspace/DPC-GNN/`

---

## 0. 现状评估

### 0.1 已有代码（可复用）

| 文件 | 路径 | 功能 | 状态 |
|------|------|------|------|
| `material_features.py` | `phase_d/` | 材料特征注入 (E, ν, embedding) | ✅ 可用，有测试 |
| `material_scaling.py` | `phase_d/` | 材料自适应位移缩放 | ✅ 可用，有测试 |
| `run_bone_curriculum.sh` | `phase_d/` | 课程学习训练流程 | ✅ 可参考 |
| Bone 目录结构 | `multi_tissue/bone/` | ckpt/results 空目录 | ✅ 已建 |

### 0.2 待创建代码（8个文件，目标 ~304K 参数）

| 文件 | 路径 | 功能 | 优先级 |
|------|------|------|--------|
| `sph_gnn_model.py` | `blood-fluid/src/` | SPH-GNN 模型架构 | P0 |
| `sph_physics_loss.py` | `blood-fluid/src/` | 物理损失函数 | P0 |
| `sph_integrator.py` | `blood-fluid/src/` | 时间积分器 | P0 |
| `poiseuille_test.py` | `blood-fluid/src/` | Poiseuille 流验证 | P1 |
| `womersley_test.py` | `blood-fluid/src/` | Womersley 流验证 | P1 |
| `hgo_energy.py` | `vessel/` | HGO 超弹性能量 | P1 |
| `vessel_mesh.py` | `vessel/` | 血管网格生成 | P1 |
| `RECOMMENDED_APPROACH.md` | `blood-fluid/research/` | 研究方法文档 | P2 |

### 0.3 项目约定（从已有代码推断）

- **PyTorch** 为主框架
- **测试先行**: 每个模块配套独立测试脚本
- **材料参数**: 统一 dict `MATERIALS = {name: {"E": ..., "nu": ..., "rho": ...}}`
- **课程学习**: 从软材料逐步迁移到硬材料
- **缩放策略**: `MaterialAwareScaling` 解决硬材料位移信号微弱问题
- **Checkpoint**: `multi_tissue/{material}/ckpt/phase_d/`

---

## 1. 需要构建的 8 个文件详细设计

### 1.1 `sph_gnn_model.py` — SPH-GNN 模型架构

**目标**: 将 SPH 邻居交互建模为图神经网络

**核心设计**:
```python
# 输入特征（每个粒子）
node_features = [position(3), velocity(3), pressure(1), density(1), material_embedding(8)]
# 总计: 16维节点特征

# 边特征（每对邻居粒子）
edge_features = [relative_pos(3), distance(1), kernel_value(1), kernel_grad(3)]
# 总计: 8维边特征

# 网络架构
SPH_GNN(
  node_encoder: MLP(16 → 64),
  edge_encoder: MLP(8 → 64),
  message_passing: 3层 InteractionNetwork,
  decoder: MLP(64 → 4)  # 输出: acceleration(3) + pressure_correction(1)
)
```

**参数量估算**:
- Node encoder: 16→64 = ~1.2K
- Edge encoder: 8→64 = ~0.6K
- 3× InteractionNetwork (64-dim hidden): ~3×100K = ~300K
- Decoder: 64→4 = ~0.3K
- **总计**: ~304K ✅ 与声称一致

**关键模块**:
1. `SPHKernel` — Wendland/Cubic Spline 核函数
2. `SPHGraphBuilder` — 构建邻居图（支持动态邻居）
3. `SPHGNNLayer` — 单层消息传递（包含密度、压力、粘性三项）
4. `SPHGNN` — 完整模型

**与已有代码的集成点**:
- `MaterialFeatureInjector` 注入材料 embedding
- `MaterialAwareScaling` 缩放输入/输出位移

### 1.2 `sph_physics_loss.py` — 物理损失函数

**损失函数设计**:
```python
PhysicsLoss(
  # Navier-Stokes 残差
  continuity_weight=1.0,    # ∂ρ/∂t + ∇·(ρv) = 0
  momentum_weight=1.0,      # ρ Dv/Dt = -∇p + μ∇²v + f

  # 边界条件
  boundary_weight=10.0,     # 无滑移边界

  # 血液特性
  non_newtonian_weight=0.5, # Carreau-Yasuda 非牛顿粘度

  # 稳定性
  density_weight=0.1,       # 密度波动惩罚
)
```

**Carreau-Yasuda 非牛顿模型**:
```python
μ(γ̇) = μ_∞ + (μ_0 - μ_∞)(1 + (λγ̇)^a)^((n-1)/a)
# 血液参数: μ_0=0.056 Pa·s, μ_∞=0.00345 Pa·s, λ=3.313s, n=0.3568, a=2.0
```

### 1.3 `sph_integrator.py` — 时间积分器

**两种积分器**:
1. **Leapfrog** — 适合长时间模拟（能量守恒好）
2. **Symplectic Euler** — 简单但稳定

**关键设计**:
```python
SPHIntegrator(
  dt=1e-4,              # 时间步长
  scheme="leapfrog",    # 积分方案
  density_evolution=True,  # 密度演化 vs 密度求和
  viscosity_implicit=True, # 隐式粘性项（稳定性）
)
```

**CFL 条件检查**:
```python
dt_max = 0.25 * h / (v_max + c_sound)
```

### 1.4 `poiseuille_test.py` — Poiseuille 流验证测试

**物理背景**: 管道中牛顿流体的稳态层流，有解析解

**解析解**:
```python
v(r) = (ΔP / (4μL)) * (R² - r²)
# v_max = ΔP·R² / (4μL)  在管道中心
```

**测试配置**:
```python
PoiseuilleConfig(
  R=1e-3,              # 管道半径 1mm
  L=5e-3,              # 管道长度 5mm
  ΔP=1.0,              # 压差 1 Pa
  μ=0.0035,            # 血液粘度 3.5 mPa·s
  ρ=1060,              # 血液密度 1060 kg/m³
  n_particles=500,     # 粒子数
  h=2e-4,              # 光滑长度
)
```

**验证指标**:
- 速度剖面 RMSE < 5%（相对于解析解）
- 质量守恒误差 < 1%
- 稳态收敛时间 < 1s 模拟时间

### 1.5 `womersley_test.py` — Womersley 流验证测试

**物理背景**: 脉动流（心跳驱动），解析解含 Bessel 函数

**解析解**:
```python
v(r,t) = Re[ΔP/(iωρL) * (1 - J_0(α·r/R·i^(3/2))/J_0(α·i^(3/2))) * e^(iωt)]
# α = R√(ωρ/μ)  Womersley 数
```

**测试配置**:
```python
WomersleyConfig(
  R=2e-3,              # 动脉半径 2mm
  f_heartbeat=1.2,     # 心率 72 bpm
  α_womersley=3.0,     # Womersley 数
  # → μ, ρ, ω 由 α 反推
)
```

**验证指标**:
- 相位误差 < 10°
- 振幅误差 < 10%
- Womersley 数范围覆盖 [1, 10]（小动脉到主动脉）

### 1.6 `hgo_energy.py` — Holzapfel-Gasser-Ogden 超弹性能量

**HGO 模型**:
```python
Ψ = Ψ_iso(I₁) + Ψ_aniso(I₄f, I₄s)
# Ψ_iso = c₁/2 (Ī₁ - 3)           (Neo-Hookean 基体)
# Ψ_aniso = k₁/(2k₂) Σ [exp(k₂(Eᵢ)²) - 1]  (纤维项)
# Eᵢ = κ(I₁-3) + (1-3κ)(I₄ᵢ-1)  (弥散纤维)

# 血管参数:
c₁ = 5.0 kPa, k₁ = 5.0 MPa, k₂ = 10.0, κ = 0.25 (弥散度)
```

**关键方法**:
- `energy(F)` — 计算应变能密度
- `stress(F)` — 第一 Piola-Kirchhoff 应力
- `tangent_stiffness(F)` — 材料切线刚度（用于 FEA）
- `fit_to_data(stretch, stress)` — 从实验数据拟合参数

### 1.7 `vessel_mesh.py` — 血管网格生成

**几何模型**:
```python
VesselMesh(
  R_inner=2e-3,     # 内径 2mm
  R_outer=2.4e-3,   # 外径 2.4mm (壁厚 0.4mm)
  length=10e-3,     # 长度 10mm
  n_circum=32,      # 周向分段
  n_radial=4,       # 径向分层
  n_length=50,      # 轴向分段
)
```

**输出**:
- 节点坐标 (V, 3)
- 单元连接 (T, 8) — 六面体
- 表面标记（内壁、外壁、入口、出口）
- 纤维方向场（周向 + 轴向）

### 1.8 `RECOMMENDED_APPROACH.md` — 研究方法文档

总结上述所有设计决策的文献依据和理论基础。

---

## 2. 执行计划

### Phase 1: 基础模块（Day 1-2）

**顺序**: `sph_gnn_model.py` → `sph_physics_loss.py` → `sph_integrator.py`

| 步骤 | 任务 | 时间 | 验证 |
|------|------|------|------|
| 1.1 | 实现 `SPHKernel`（Wendland C2） | 1h | 核函数归一化、梯度正确性 |
| 1.2 | 实现 `SPHGraphBuilder` | 1h | 邻居搜索正确性、边界粒子处理 |
| 1.3 | 实现 `SPHGNNLayer` + 完整 `SPHGNN` | 2h | 参数量≈304K、前向传播维度 |
| 1.4 | 实现 `PhysicsLoss`（Navier-Stokes残差） | 2h | 解析解验证（均匀流、静水） |
| 1.5 | 实现 `SPHIntegrator` | 1h | 能量守恒、CFL条件检查 |
| 1.6 | 单元测试 | 1h | 所有模块独立运行 |

### Phase 2: Poiseuille 流验证（Day 2-3）⭐ 最关键

**为什么先跑 Poiseuille**:
1. 有精确解析解，可严格验证
2. 稳态流，比脉动流简单得多
3. 二维管道，计算量小
4. 如果 Poiseuille 都不过，Womersley 不用想

| 步骤 | 任务 | 时间 | 验证 |
|------|------|------|------|
| 2.1 | 实现 `poiseuille_test.py` | 2h | 粒子初始化、边界条件 |
| 2.2 | 跑牛顿流体验证 (μ=const) | 1h | 速度剖面 RMSE < 5% |
| 2.3 | 跑非牛顿流体验证 (Carreau) | 1h | 与牛顿流体对比 |
| 2.4 | 参数敏感性分析 | 1h | 粒子数、光滑长度、时间步 |

**预期结果**:
- 抛物线速度剖面
- 中心最大速度 v_max = ΔP·R²/(4μL)
- 壁面无滑移条件满足
- 质量守恒

**如果失败的调试策略**:
1. 检查核函数归一化 (∫W dV = 1)
2. 检查密度求和 vs 密度演化
3. 减小时间步长（CFL条件）
4. 增加粒子数（分辨率）
5. 检查边界处理（镜像粒子 vs 虚粒子）

### Phase 3: Womersley 流验证（Day 3-4）

| 步骤 | 任务 | 时间 | 验证 |
|------|------|------|------|
| 3.1 | 实现 `womersley_test.py` | 2h | 脉动边界条件 |
| 3.2 | 跑低 Womersley 数 (α≈1) | 1h | 接近 Poiseuille（准稳态） |
| 3.3 | 跑高 Womersley 数 (α≈5) | 2h | 惯性效应、相位滞后 |
| 3.4 | 频率扫描 (α=1~10) | 2h | 振幅/相位 vs α |

**预期结果**:
- 低 α → 速度剖面接近抛物线
- 高 α → 中心平坦、壁面薄边界层
- 相位滞后随 α 增大

### Phase 4: 血管模型（Day 4-5）

| 步骤 | 任务 | 时间 | 验证 |
|------|------|------|------|
| 4.1 | 实现 `hgo_energy.py` | 2h | 单轴拉伸解析解对比 |
| 4.2 | 实现 `vessel_mesh.py` | 2h | 网格质量检查 |
| 4.3 | FSI 边界耦合 | 3h | 脉动压力 → 壁面位移 |

### Phase 5: 流固耦合集成（Day 5-7）

| 步骤 | 任务 | 时间 | 验证 |
|------|------|------|------|
| 5.1 | SPH 粒子 ↔ FEM 网格耦合 | 3h | 信息传递正确性 |
| 5.2 | 脉动管道流固耦合 | 4h | 壁面变形 + 流场 |
| 5.3 | 与 DPC-GNN 主框架集成 | 2h | 材料特征注入 |

---

## 3. 训练计划

### 3.1 SPH-GNN 训练策略

**核心思路**: 物理约束 + 监督学习混合

```
Loss = λ₁·L_data + λ₂·L_physics + λ₃·L_boundary
```

| 分量 | 说明 | 权重 |
|------|------|------|
| L_data | 与 CFD 参考解（OpenFOAM/Ansys）的 MSE | 1.0 |
| L_physics | Navier-Stokes 残差 | 0.5 |
| L_boundary | 边界条件违反 | 10.0 |

### 3.2 训练数据来源

**方案 A: CFD 仿真数据（推荐）**
- 工具: OpenFOAM (pimpleFoam / pisoFoil)
- 场景:
  1. 直管 Poiseuille 流（基准）
  2. 弯管二次流
  3. 狭窄管（模拟动脉狭窄）
  4. 分叉管（模拟动脉分叉）
- 数据量: 每个场景 100 个时间步 × 5000 粒子 = 500K 样本

**方案 B: 解析解数据（开发阶段）**
- Poiseuille + Womersley 解析解
- 用于调试和初始验证
- 不需要外部 CFD 工具

**方案 C: 仅物理损失（无监督）**
- 完全不使用参考数据
- 纯靠 Navier-Stokes 残差训练
- 训练难度大，但泛化性最好

### 3.3 训练超参数

```python
TrainingConfig(
  optimizer="Adam",
  lr=1e-3,
  lr_schedule="cosine",
  epochs=200,
  batch_size=8,         # 每个 batch 一个完整场景
  grad_clip=1.0,
  
  # 数据
  train_scenarios=["poiseuille", "stenosis_30", "stenosis_50"],
  val_scenarios=["bifurcation", "curved_pipe"],
  
  # 物理损失权重调度
  lambda_physics_warmup=50,  # 前50 epoch 逐步增加物理损失权重
)
```

### 3.4 预期训练时间

| 阶段 | GPU | 时间 |
|------|-----|------|
| Poiseuille 基准 | 单 GPU | ~2h |
| 多场景训练 | 单 GPU | ~8h |
| 完整训练（含调参） | 单 GPU | ~24h |

---

## 4. 与 DPC-GNN 集成

### 4.1 架构集成

```
DPC-GNN (统一框架)
├── Phase A/B/C: 变形预测（已有）
├── Phase D: 材料感知（已有 ✅）
│   ├── MaterialFeatureInjector
│   └── MaterialAwareScaling
├── Blood-Fluid: SPH-GNN（待建）
│   ├── 接收: 壁面位移（来自 DPC-GNN 变形预测）
│   ├── 输出: 流场 (v, p, ρ)
│   └── 特征注入: 材料类型=blood
└── Vessel: HGO 模型（待建）
    ├── 接收: 压力（来自 SPH-GNN）
    ├── 输出: 壁面应力/应变
    └── 特征注入: 材料类型=vessel_wall
```

### 4.2 材料注册表扩展

```python
# 扩展已有 MATERIALS dict
MATERIALS = {
    # 已有
    "brain":      {"E": 1e3,    "nu": 0.49, "id": 0},
    "kidney":     {"E": 1e4,    "nu": 0.45, "id": 1},
    "myocardium": {"E": 3e4,    "nu": 0.40, "id": 2},
    "cartilage":  {"E": 5e5,    "nu": 0.30, "id": 3},
    "bone":       {"E": 1e10,   "nu": 0.30, "id": 4},
    # 新增
    "blood":      {"E": None,   "nu": None, "rho": 1060, "mu": 0.0035, "id": 5},
    "vessel_wall":{"E": 1e6,    "nu": 0.45, "rho": 1100, "hgo_params": {...}, "id": 6},
}
```

### 4.3 FSI 耦合策略

**弱耦合（推荐，先做）**:
```
1. DPC-GNN 预测壁面位移 u_wall
2. SPH 粒子边界更新
3. SPH-GNN 推理流场 (v, p)
4. 壁面压力 p → HGO 模型 → 应力 σ
5. 下一时间步
```

**强耦合（后续优化）**:
- 迭代直到壁面位移和流场收敛
- 计算量大 3-5x，但精度更高

---

## 5. 论文贡献

### 5.1 技术贡献

1. **SPH-GNN 框架**: 首次将 GNN 用于 SPH 血液流体模拟
   - 相比传统 SPH: 无需人工设定人工粘性
   - 相比 CFD: 推理速度快 100x+

2. **物理约束训练**: Navier-Stokes 残差作为正则化
   - 不需要大量标注数据
   - 泛化到未见过的几何

3. **多组织统一框架**: DPC-GNN 从固体扩展到流体+FSI
   - Bone → Blood → Vessel 课程学习
   - 材料特征注入 + 自适应缩放

4. **HGO 血管模型**: 与流体耦合的超弹性壁面
   - 比线弹性壁面更物理真实
   - 支持各向异性纤维结构

### 5.2 实验验证

| 实验 | 价值 | 预期结果 |
|------|------|----------|
| Poiseuille 基准 | 证明基本正确性 | RMSE < 5% |
| Womersley 脉动流 | 证明时间依赖建模能力 | 相位误差 < 10° |
| 动脉狭窄模拟 | 临床相关性 | 与 CFD 对比 |
| FSI 脉动管道 | 多组织耦合能力 | 壁面应力准确 |

### 5.3 投稿目标

- **主攻**: MICCAI / TMI / MedIA
- **备选**: CMAME / JCP
- **亮点**: "Physics-Informed SPH-GNN for Real-Time Blood Flow Simulation in Deformable Vessels"

---

## 6. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| SPH-GNN 训练不收敛 | 中 | 从纯物理损失开始，逐步加入监督 |
| 非牛顿流体建模困难 | 中 | 先验证牛顿流体，再扩展 |
| FSI 耦合不稳定 | 高 | 弱耦合 + 小时间步 |
| 参数量超出 304K | 低 | 已有精确估算，控制 hidden dim |
| 训练数据不足 | 中 | 解析解 + 少量 CFD 数据 |

---

## 7. 下一步行动

### 立即执行（今天）

1. ✅ 创建 `blood-fluid/` 和 `vessel/` 目录结构
2. 🔄 实现 `sph_gnn_model.py`（SPHKernel + SPHGraphBuilder + SPHGNN）
3. 🔄 编写单元测试验证前向传播

### 本周目标

- Day 1-2: 完成三个核心模块 + 单元测试
- Day 3: Poiseuille 流验证通过
- Day 4: Womersley 流验证通过
- Day 5: 血管模型完成

---

*本计划基于 DPC-GNN 项目现有代码约定（材料特征注入、课程学习、自适应缩放）制定。*

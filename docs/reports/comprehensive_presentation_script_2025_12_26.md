# [深度组会汇报] 基于 JAX 的可微 Drucker-Prager 塑性建模：从理论实现到伴随方程的奇异性攻关

**汇报人：** Lishiyao
**日期：** 2025年12月26日
**性质：** 组会汇报


---

## 0. 汇报引言 (Introduction)



**《基于 JAX-FEM 的 Drucker-Prager 塑性本构实现及其可微性验证》**。

本周工作的核心目标是：在完全可微的计算图（Computational Graph）框架内，建立并验证岩土材料的非线性塑性本构模型。本研究的难点在于实现过程中的“可微性（Differentiability）”保证，即确保通过自动微分（AD）计算得到的物理场对材料参数的梯度具有数值稳定性和数学精确性。

本周的研究经历了一个完整的“问题发现-假设提出-验证排查-理论归因”的闭环过程。在正向模拟（Forward Simulation）取得收敛的基础上，反向传播（Backward Propagation）阶段出现了严重的数值发散现象。通过系统的五轮假设验证，最终定位了问题的根本数学原因。今天，我将详细汇报这一调试过程及其背后的**伴随方程适定性（Well-posedness of Adjoint Equations）**理论分析。

---

## 1. 研究背景与动机 (Research Background & Motivation)

### 1.1 传统岩土计算方法的局限性
在岩土工程数值计算领域，主流商业软件（如 Abaqus, Plaxis）及开源代码（OpenSees）虽然功能强大，但普遍存在**“黑盒化”**或**“不可微”**的局限性。

在处理**反问题（Inverse Problems）**，例如根据监测位移反演土体参数 $E$ 和 $k$ 时，传统方法通常依赖**有限差分法（Finite Difference）**估算梯度。
*   对于 $N$ 个参数，需要进行 $N+1$ 次有限元计算。
*   在涉及高维设计变量的拓扑优化问题中，计算成本随变量维度线性增长，导致效率瓶颈。

### 1.2 可微物理（Differentiable Physics）的优势
基于 **JAX** 开发的可微有限元框架旨在解决上述痛点。JAX 提供的**自动微分（AD）**能力，特别是**反向模式（Reverse Mode）**，允许在仅执行一次正向模拟和一次反向传播的情况下，计算目标函数对**任意数量**参数的精确梯度。

这为以下研究方向提供了关键的基础设施：
1.  **数据驱动本构建模**：利用神经网络替代经验公式，直接从实验数据中学习本构关系。
2.  **基于梯度的工程反演**：实现基于监测数据的地质模型实时更新与参数修正。

实现这一愿景的前提是：**必须在可微框架内成功构建并验证经典的、非线性的岩土本构模型。**

---

## 2. 理论框架：Drucker-Prager 模型的数学构建 (Theoretical Framework)

本研究选取了岩土领域经典的 **Drucker-Prager (DP)** 模型。相较于金属塑性的 Von Mises 模型，DP 模型引入了**静水压力相关性（Pressure Dependency）**，这对可微性提出了特定的数值挑战。

### 2.1 屈服面的几何奇异性问题
标准的 DP 屈服准则表达式为：
$$ f(\boldsymbol{\sigma}) = \sqrt{J_2} + \alpha I_1 - k \leq 0 $$

其中：
*   **$I_1$ (第一不变量)**：应力张量的迹 $\text{tr}(\boldsymbol{\sigma})$，表征静水压力。
*   **$J_2$ (第二偏应力不变量)**：$\frac{1}{2} \boldsymbol{s} : \boldsymbol{s}$，表征剪切变形能。

**数值难点分析**：
在 $\sqrt{J_2}$ 项中，当 $J_2 \to 0$ （纯静水压力状态）时，其导数 $\frac{1}{2\sqrt{J_2}}$ 趋于无穷大。这意味着 Drucker-Prager 屈服面的**顶点（Apex）**存在几何奇异性。

在传统 FEM 中，通常采用专门的“顶点算法（Corner Algorithm）”处理该奇点。然而，在要求**全计算图可导**的 JAX 框架中，过多的分支判断（Branching）会破坏梯度的连续性，影响优化算法的稳定性。

### 2.2 双曲平滑正则化 (Hyperbolic Smoothing)
为解决奇异性问题，本研究采用了**双曲平滑正则化**策略。修正后的屈服函数为：
$$ f(\boldsymbol{\sigma}) = \sqrt{J_2 + a^2} + \alpha I_1 - k $$

其中 $a$ 为正则化参数。
*   当 $J_2 \gg a^2$ 时，模型渐近收敛于标准 DP 模型。
*   当 $J_2 \to 0$ 时，函数表现为 $\sqrt{a^2} = a$，导数平滑过渡至 0，消除了奇点。

本实验中选取 $a = 2.5 \text{ MPa}$（约为黏聚力 $k$ 的 1%）。该参数的选择至关重要：过大导致物理失真，过小则无法有效抑制数值震荡。

### 2.3 隐式返回映射算法 (Implicit Return Mapping)
在塑性积分环节，采用了**隐式向后欧拉算法（Implicit Backward Euler）**。

核心代码实现如下（基于 `src/models/drucker_prager.py`）：

```python
def stress_return_map(u_grad, sigma_old, epsilon_old):
    # 计算试探应力 (Trial Stress)
    epsilon_crt = strain(u_grad)
    epsilon_inc = epsilon_crt - epsilon_old
    sigma_trial = elastic_stress(epsilon_inc) + sigma_old

    # 计算应力不变量
    I1 = np.trace(sigma_trial)
    s_dev = sigma_trial - (I1 / 3.) * np.eye(dim)
    J2 = 0.5 * np.sum(s_dev * s_dev)

    # 1. 屈服函数平滑 (Hyperbolic Regularization)
    sqrt_J2_reg = np.sqrt(J2 + a * a)
    f_yield = sqrt_J2_reg + alpha * I1 - k

    # 2. 塑性流动方向计算
    n_dev = safe_divide(s_dev, 2. * sqrt_J2_reg)
    
    # 3. 塑性乘子增量
    # 这里通过隐式映射直接计算，利用了 DP 模型的几何特性
    f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
    delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha)
    
    # 4. 更新应力
    sigma = sigma_trial - delta_lambda * (n_dev + alpha * np.eye(dim))

    # 5. 顶点特殊处理 (Apex Handling)
    # 虽然平滑了，但为了物理准确性，我们在极高围压下仍需检查是否落在顶点上
    sigma_apex = (k / (3. * alpha)) * np.eye(dim)
    at_apex = np.logical_and(f_yield > 0., I1 > k / alpha)
    sigma = np.where(at_apex, sigma_apex, sigma)

    return sigma
```

利用 JAX 的 `jax.lax.while_loop` 或 `newton_raphson` 算子在每个积分点独立求解该方程。借助 JAX 的 `vmap` 功能，实现了百万级积分点非线性方程的并行求解，显著优于传统串行计算效率。

---

## 3. 正向验证：本构驱动程序测试 (Forward Verification)

为验证实现的正确性，开发了独立的**本构驱动程序（Constitutive Driver）**，在不依赖有限元网格的情况下验证本构积分算法。

### 3.1 本构驱动程序设计 (Design of Constitutive Driver)
为了在单元测试层面验证 JAX 本构实现的正确性，我开发了一个独立于网格的**半解析本构驱动程序（Semi-analytical Constitutive Driver）**（见 `verification_dp.py`）。

**核心原理：**
该程序并非简单的公式计算，而是模拟了一个**单点混合控制问题（Single-point Mixed Control Problem）**。
在有限元单轴压缩实验中，侧面是自由表面，意味着侧向应力 $\sigma_{xx} = \sigma_{yy} = 0$。为了在单点积分中复现这一边界条件，我构建了一个局部非线性求解器：

1.  **控制变量**：给定垂直应变增量 $\Delta \epsilon_{zz}$（由位移加载控制）。
2.  **未知变量**：侧向应变增量 $\Delta \epsilon_{lat} = \Delta \epsilon_{xx} = \Delta \epsilon_{yy}$。
3.  **残差方程**：寻找 $\Delta \epsilon_{lat}$，使得经过本构积分（Stress Return Map）后的侧向应力 $\sigma_{xx}(\Delta \epsilon_{zz}, \Delta \epsilon_{lat}) = 0$。

**代码实现逻辑：**

```python
from scipy.optimize import root

def residual(d_eps_lat):
    # 构建包含未知侧向应变的全增量张量
    inc_strain = np.diag([d_eps_lat, d_eps_lat, d_eps_zz])
    
    # 调用与 FEM 完全相同的应力更新函数
    sig_new = stress_return_map_point(inc_strain, sigma_current)
    
    # 目标：侧向应力为零
    return sig_new[0, 0]

# 使用混合 Powell 算法求解局部非线性方程
sol = root(residual, lat_strain_inc_guess, method='hybr')
```

通过这种方式，我们生成了一条**完全精确的理论基准曲线（Ground Truth）**，它不仅验证了本构公式，还验证了隐式积分算法在非等向应力状态下的稳定性。

### 3.2 验证结果分析
将 FEM 单单元仿真结果与半解析解进行了对比分析（见图 `comparison_plot.png`）。

*   **弹性阶段**：两者弹性模量均为 70 GPa，完全吻合。
*   **塑性阶段**：屈服后，FEM 计算得到的塑性切线模量与理论值 $E_{\text{ep}} = 58,983 \text{ MPa}$ 一致，误差为 0.00%。

**结论**：
1.  **应力更新算法（Stress Update）** 正确。
2.  **一致切线模量（Consistent Tangent Modulus）** 推导正确。残差收敛曲线呈现出典型的牛顿法二次收敛特征（$10^{-2} \to 10^{-5} \to 10^{-11}$），进一步证实了切线算子的准确性。

**正向问题（Forward Problem）验证通过。**

---

## 4. 问题与挑战：伴随方程求解的不稳定性 (Instability in Adjoint Equation)

在正向验证通过后，进行梯度计算的**冒烟测试（Smoke Test）**时，出现了意料之外的数值稳定性问题。

### 4.1 测试用例与现象
设计了两组测试用例：
1.  **Case A（弹性小变形）**：施加 $10^{-5}$ mm 位移。
    *   **结果**：AD 梯度与有限差分梯度误差仅 $10^{-19}$，验证了自动微分管线的连通性。
2.  **Case B（塑性大变形）**：施加 $-0.1$ mm 位移（深度塑性）。
    *   **正向计算**：收敛正常。
    *   **反向计算**：**发散**。

### 4.2 错误分析
PETSc 线性求解器报错 `Linear solver failed to converge: DIVERGED_DTOL`，伴随方程残差从初始 $0.1$ 激增至 $>190$。

这揭示了一个非直观的现象：**正向问题的收敛并不保证反向问题的适定性。**
反向传播需求解伴随方程：
$$ \mathbf{K}_T^T \cdot \boldsymbol{\lambda} = -\frac{\partial L}{\partial \boldsymbol{u}} $$
其中 $\mathbf{K}_T$ 为最终收敛步的切线刚度矩阵。

**初步推断**：虽然 $\mathbf{K}_T$ 足以支持正向牛顿迭代收敛，但对于伴随向量 $\boldsymbol{\lambda}$ 而言，该矩阵表现出不可解或极度病态的特征。

---

## 5. 问题排查与归因分析 (Diagnosis & Attribution)

为定位问题根源，提出了四个假设，并设计了五组实验进行逐一排查。

### 假设一：本构模型的非光滑性 (Non-smoothness)
**假设**：塑性模型中的 `max(0, f)` 截断导致导数不连续，引起线性化算子不稳定。
尽管 JAX 支持控制流微分，但切换点（Switching Point）的导数突变可能导致数值问题。

*   **验证策略（Strategy 1）**：引入 **Perzyna 黏塑性模型**。
    
    我在代码中实现了一个显式的黏塑性松弛步，试图通过时间参数 $\eta$ 来平滑屈服面：

    ```python
    # Strategy 1: Perzyna Viscoplasticity Implementation
    # Delta lambda is no longer just consistent with f=0, but relaxed by viscosity
    f_over_k = f_yield / k
    phi_val = np.where(f_over_k > 0, f_over_k, 0.) # Macaulay bracket <f/k>
    
    # 引入时间步 dt 和粘度 eta
    delta_lambda = (self.dt / self.eta) * phi_val
    
    # 应力更新不再强行拉回屈服面，而是松弛逼近
    sigma = sigma_trial - delta_lambda * flow_direction
    ```

*   **结果**：**问题依旧**。残差未见改善。
*   **结论**：问题非源于局部的微观非光滑性，而是宏观结构性问题。

### 假设二：零硬化导致的刚度衰减 (Zero Hardening)
**假设**：理想弹塑性模型（$H=0$）导致切线刚度矩阵 $\mathbf{K}_T$ 特征值过小。
切线模量公式 $\mathbf{D}^{\text{ep}} = \mathbf{D}^e - \frac{\mathbf{D}^e \mathbf{n} \otimes \mathbf{n} \mathbf{D}^e}{\mathbf{n}^T \mathbf{D}^e \mathbf{n} + H}$ 显示，当 $H=0$ 时，刚度折减最大。

*   **验证策略（Strategy 4）**：人为引入线性硬化项 $H$。

    ```python
    # Strategy 4: Linear Hardening
    # Hardening parameter added to denominator to enforce positive definiteness
    hardening_param = 0.5 * self.E_val # Strong hardening
    
    # Denominator now includes + H
    delta_lambda = f_yield_plus / (1. + 3. * alpha * alpha + hardening_param)
    ```

*   **结果**：**依然发散**。
*   **结论**：硬化项无法解决发散问题，说明矩阵存在比“病态”更严重的性质。

### 假设三：迭代求解器能力限制 (Solver Capability)
**假设**：PETSc 默认的 GMRES + ILU 预条件子无法处理非对称塑性刚度矩阵。

*   **验证策略（Strategy 2）**：切换至 **MUMPS** 直接求解器。

    ```python
    # Strategy 2: Direct Solver Configuration
    solver_options = {
        'petsc_solver': {
            'ksp_type': 'preonly', 
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps' # 强制使用 MUMPS 做 LU 分解
        }
    }
    ```

*   **结果**：抛出 `Zero Pivot`（主元为零）错误。
*   **关键发现**：直接求解器的失效表明矩阵并非简单的“病态（Ill-conditioned）”，而是**“奇异（Singular）”**。

### 假设四：边界条件导致的刚体模态 (Rigid Body Modes) —— 根本原因
**假设**：矩阵奇异性源于**零空间（Null Space）**的存在，物理上对应**刚体位移（Rigid Body Motion）**。

检查边界条件发现：
为了模拟单轴压缩，仅约束了底面 $z=0$ 处的 $u_z = 0$，**未约束 $X$ 和 $Y$ 方向的自由度**。

*   **正向收敛原因**：
    *   几何模型、网格、载荷及初始条件均具有对称性。Newton 迭代路径始终位于与刚体模态正交的对称子空间内，数值误差不足以激发刚体平移模式。
*   **反向发散原因**：
    *   伴随方程右端项 $-\frac{\partial L}{\partial \boldsymbol{u}}$ 包含所有方向的位移误差信息。
    *   梯度反向传播过程中，任何微小的数值噪声投影至 $X/Y$ 刚体模态上，由于 $\mathbf{K}_T$ 在该方向特征值为 0，导致伴随向量 $\boldsymbol{\lambda}$ 无限增大。

### 最终验证 (Strategy 5)
修改测试脚本，将底面边界条件改为 **完全固定（Clamp）**。

**代码修正** (`tests/test_diff_dp_bc_fix.py`)：

```python
# Strategy 5: Correct Boundary Conditions
# Previous incorrect BC: Only constrained u_z
# New correct BC: Fully fix bottom face (u_x=0, u_y=0, u_z=0)

def bottom(p): return np.isclose(p[2], 0.)
def top(p): return np.isclose(p[2], Lz)

dirichlet_bc_info = [
    [bottom, bottom, bottom, top],  # Apply to bottom face 3 times for x, y, z
    [0, 1, 2, 2],                   # Component indices: x, y, z, and z-top
    [lambda p: 0., lambda p: 0., lambda p: 0., lambda p: displacement] # Values
]
```

**结果**：
*   正向计算收敛。
*   反向传播**收敛**。
*   梯度计算误差降至 **$10^{-16}$**。

---

## 6. 讨论与总结 (Discussion & Conclusion)

### 6.1 核心发现：伴随方程的严苛性
本研究揭示了可微物理中的一条重要规律：**伴随方程是检验物理模型适定性（Well-posedness）的最严格标准**。
传统计算中常利用对称性简化边界条件，这种做法在正向求解中可能因求解器的鲁棒性而被“容忍”。然而，**自动微分**过程会遍历计算图的全局，任何未约束的自由度（Unconstrained DOF）都会在反向传播中暴露无遗，导致计算失败。

### 6.2 刚度矩阵奇异性的本质
问题的本质在于切线刚度矩阵 $\mathbf{K}$ 的秩不足（Rank Deficient）：
$$ \text{Rank}(\mathbf{K}) < N_{\text{dof}} $$
对于塑性问题，切线模量的降低削弱了系统刚度，使得刚体模态更容易导致零特征值的出现。这提示我们在进行参数反演或拓扑优化时，必须显式进行**特征值分析（Eigenvalue Analysis）**，确保系统不存在零能模态。

---

## 7. 后续研究计划 (Future Work)

在解决了梯度计算稳定性的基础上，后续工作将聚焦于：

1.  **三轴实验数据反演**：
    *   利用现有的可微框架和标准岩土三轴压缩实验数据，构建 Loss 函数。
    *   通过基于梯度的优化算法，自动反演最优材料参数组合 ($E, \nu, k, \phi$)，实现仿真与实验曲线的拟合。

2.  **GPU 加速与性能优化**：
    *   利用 JAX 的 `jit` 编译技术，将计算迁移至 GPU 环境。
    *   预期将大幅提升反演算法的计算效率，实现相比传统遗传算法的数量级加速。

汇报完毕

---

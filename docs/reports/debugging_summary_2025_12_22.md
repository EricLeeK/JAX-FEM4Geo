# Drucker-Prager 模型可微分性调试总结报告

**日期：** 2025年12月22日
**环境：** WSL (Ubuntu 22.04), JAX-FEM, CUDA 12.x
**核心结论：** 解决了大位移下伴随方程求解不收敛的问题，确定根因为边界条件不足引起的刚体位移。

---

## 1. 问题背景与现象

在对 `src/models/drucker_prager.py` 及其对应的 JAX-FEM 封装进行微分验证时，使用 `jax.value_and_grad` 计算 Loss 对材料参数 ($E, k$) 的梯度。

### 1.1 观测到的故障
*   **弹性/微塑性区 (-1e-5 to -5e-4)**：梯度验证成功，相对误差 < 1e-10。
*   **显著塑性区 (-0.001 及以上)**：
    *   前向模拟 (Forward) 正常收敛，残差 $\sim 10^{-13}$。
    *   反向传播 (Backward/Adjoint) 阶段，PETSc 线性求解器报错：`AssertionError: PETSc linear solver failed to converge, err = 0.13...`。
    *   位移越大，伴随方程的残差越大（从 0.13 增加到 190+）。

---

## 2. 技术方案与实验记录

为了定位问题，我们实施并对比了以下五种策略。对应的测试脚本均位于 `tests/` 目录下。

### 策略 1：Perzyna 黏塑性 (Viscoplasticity)
*   **脚本**：`tests/test_diff_dp_strategy1_visco.py`
*   **实现**：将率无关塑性公式修改为：$\Delta \lambda = \frac{\Delta t}{\eta} \langle \frac{f}{k} \rangle$。
*   **预期**：通过引入粘性项来平滑屈服面的转折点，改善 Jacobian 的条件数。
*   **结果**：**失败**。虽然物理上更光滑，但未解决伴随方程发散问题。

### 策略 2：直接求解器优化 (Direct Solver)
*   **脚本**：`tests/test_diff_dp_strategy2_direct.py`
*   **实现**：强制 PETSc 使用 `mumps` 或 `superlu_dist` 进行 LU 分解。
*   **预期**：利用直接求解器硬算病态矩阵。
*   **结果**：**失败**。残差依然巨大，说明矩阵本质奇异（Singular），而非简单的病态。

### 策略 3：平滑近似 (Smooth Approximation)
*   **脚本**：`tests/test_diff_dp_strategy3_smooth.py`
*   **实现**：使用 `softplus` 替换 `max(0, x)`，使用 `sigmoid` 替换逻辑判断。
*   **预期**：消除 $C^0$ 连续性带来的导数突变。
*   **结果**：**失败**。

### 策略 4：材料硬化 (Linear Hardening)
*   **脚本**：`tests/test_diff_dp_hardening.py`
*   **实现**：在回映算法的分母中引入硬化模量 $H$：$\Delta \lambda = \frac{f}{D + H}$。
*   **预期**：保证屈服后的切线模量不为零。
*   **结果**：在旧 BC 下依然**失败**。

### 策略 5：边界条件修复 (BC Fix) - **最终方案**
*   **脚本**：`tests/test_diff_dp_bc_fix.py`
*   **实现**：将原有的仅 Z 方向约束改为**完全固定底面**（约束 X, Y, Z 位移）。
*   **原理**：原模型在 X, Y 方向存在刚体平移自由度，导致刚度矩阵存在零特征值。
*   **结果**：**成功**。位移在 -0.1 下依然能够完美计算梯度。

---

## 3. 运行与验证指南

若要复现验证结果，请在 WSL 环境中运行以下命令：

```bash
# 激活环境
conda activate jax_gpu

# 运行基础修复验证 (推荐)
python tests/test_diff_dp_bc_fix.py

# 运行硬化模型验证 (更稳健)
python tests/test_diff_dp_hardening.py
```

---

## 4. 经验总结与最佳实践

### 4.1 自动微分对物理适定性的敏感性
在有限元自动微分中，**前向求解的收敛并不代表反向传播一定能行**。
*   前向求解器（尤其是 Newton 方法配合鲁棒的线性求解器）有时可以通过数值截断或对称性“躲过”奇异矩阵。
*   伴随方程求解 $K^T \lambda = f$ 对矩阵奇异性极其敏感。任何未约束的刚体位移都会导致伴随向量 $\lambda$ 发散。

### 4.2 调试建议
1.  **优先检查 BC**：确保模型在所有维度上都没有多余的自由度。
2.  **从小位移开始**：先验证线性区（弹性），再逐步增加载荷步。
3.  **区分“病态”与“奇异”**：如果直接求解器（LU）也无法给出低残差结果，优先怀疑“奇异”（BC 问题或机制失效），而非“病态”（材料突变）。
4.  **硬化作为正则化**：在进行基于梯度的优化（Inversion/Optimization）时，即使真实材料是理想塑性，在初级训练阶段引入微小的硬化项有助于数值梯度的稳定。

---
**文档状态**：已完结。所有验证脚本已保留在 `tests/` 供后续查阅。


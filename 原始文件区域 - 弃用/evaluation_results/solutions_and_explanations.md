# Solutions & Explanations: Bridging the Gaps

这份文档详细解答了在 [Evaluation Report](evaluation_report.md) 中指出的误解和盲区。

---

## 1. 核心误解修正 (Addressing Misconceptions)

### (1) 边界条件与刚体位移 (Rigid Body Motion)
**问题回顾**：为什么只约束 Z 轴位移会导致计算失败？
**技术详解**：
有限元求解的核心是解线性方程组 $\mathbf{K}\mathbf{u} = \mathbf{f}$。
刚度矩阵 $\mathbf{K}$ 必须是**满秩 (Full Rank)** 的才能求逆（或分解）。
如果物体在空间中没有任何 X 或 Y 方向的约束，它就存在**刚体模态 (Rigid Body Modes)**：
-   沿 X 平移
-   沿 Y 平移
-   绕 Z 轴旋转

这意味着有无数多组 $\mathbf{u}$ 都能满足受力平衡（因为刚体位移不产生应变能）。在数学上，这对应于 $\mathbf{K}$ 矩阵有零特征值，行列式为 0，即**奇异矩阵 (Singular Matrix)**。
**JAX-FEM (PETSc) 求解器会直接报错**，通常是 "Zero Pivot" 或不收敛。

**正确做法**：
即使是单轴压缩，也必须消除刚体位移。例如：
-   底面中心点约束 $u_x = u_y = u_z = 0$
-   底面仅约束 $u_z=0$，但增加两个侧面约束（对称边界条件）。

### (2) 数据结构 (Double Vmap: Cells & Quadrature Points)
**问题回顾**：为什么 `vmap` 要套两层？第二层是节点吗？
**技术详解**：
在非线性 FEM 中，为了消除数值闭锁（Locking）并获得准确能量积分，我们需要使用**高斯积分 (Gaussian Quadrature)**。
-   **变量存储位置**：
    -   **位移 $\mathbf{u}$**：存储在**节点 (Nodes)** 上（连续场）。
    -   **应力 $\boldsymbol{\sigma}$ & 内变量**：存储在**积分点 (Quadrature Points)** 上（不连续，在每个单元内部独立计算）。
-   **维度解析**：
    `jax.vmap(jax.vmap(strain))` 确实是两层：
    1.  **第一层 (Outer Vmap)**: 对应 **单元 (Cells/Elements)**。并行处理网格中的所有单元。
    2.  **第二层 (Inner Vmap)**: 对应 **积分点 (Quadrature Points)**。在一个单元内部，并行处理所有的积分点（例如 HEX8 单元通常有 $2 \times 2 \times 2 = 8$ 个积分点）。

这就是为什么代码里看到 `(n_cells, n_quads, ...)` 的数据结构。如果是存储在节点上，通常维度会是 `(n_nodes, ...)`，这在 FE 架构中是本质区别。

---

## 2. 知识盲区解答 (Filling Knowledge Gaps)

### (1) 为什么回映射有解析解？(Analytical Return Map)
**盲区**：分母 `1 + 3*alpha^2` 的来源。
**推导**：
Drucker-Prager 屈服准则本质上是主应力空间的一个圆锥。
-   在偏平面（Deviatoric Plane）上，它是一个圆。
-   在子午面（Meridional Plane，即 $I_1 - \sqrt{J_2}$ 平面）上，它是一条直线：$\sqrt{J_2} + \alpha I_1 - k = 0$。

这就变成了初等几何问题：**“已知平面上一点（Trial Stress），求它到一条直线的垂直投影。”**
这个几何投影显然有解析解，不需要牛顿迭代。
-   法线方向梯度是常数（关联流动）。
-   代入几何关系简单推导，就能得到那个系数 `1/(1 + 3*alpha^2)`。这本质上是直线斜率导致的一个投影缩放因子。

### (2) 弹性刚度去哪了？(Stiffness in Return Map)
**盲区**：为什么公式里没看到 $E$ 或 $K$？
**解释**：
代码中使用的其实是简化的**关联流动法则 (Associated Flow Rule)** 且隐含了投影方向与形状的关系。
严谨的**谱分解 (Spectral Decomposition)** 回映射算法中，如果材料是各向同性线弹性的，回映射确实就在几何法线上。
但确实，如果体积模量 $K$ 和剪切模量 $\mu$ 差异巨大，严格来说应该先在主应力空间做能够体现刚度差异的变换，或者直接确认该公式已经是解耦后的结果。
代码中的写法 `sigma - delta_lambda * (...)` 其实利用了 DP 模型的特殊性：它的塑性流动方向（$\partial f / \partial \sigma$）由静水压力项（$\alpha \mathbf{I}$）和偏应力项（$\mathbf{n}_{dev}$）组成，这两者正好分别对应体积变形和剪切变形，从而极其巧合地（或者说设计地）在形式上简化了。

### (3) JxW 的数学定义
**盲区**：它到底是啥？
**定义**：
$$ \int_{\Omega} f(\mathbf{x}) d\Omega \approx \sum_{e} \sum_{q} f(\mathbf{x}_q) \cdot \underbrace{\det(\mathbf{J}(\boldsymbol{\xi}_q)) \cdot w_q}_{\text{JxW}} $$
-   $w_q$：高斯积分点的**权重 (Weight)**（比如标准区间 $[-1, 1]$ 里的权重）。
-   $\det(\mathbf{J})$：**雅可比行列式 (Jacobian Determinant)**。它把“参考单元（母单元）”的微体积 $d\xi d\eta d\zeta$ 映射到“实际物理单元”的微体积 $dx dy dz$。如果单元扭曲得很厉害，这个值就会变，甚至为负（报错）。

### (4) JAX 的状态传递
**盲区**：`solver` 怎么拿到旧变量的？
**解密**：
虽然 `solver` 函数签名里没写，但 `problem` 是作为一个 Python 对象传进去的。
在 Python 中，`problem.get_tensor_map()` 返回的是一个**闭包 (Closure)** 或者绑定方法。
这个方法内部**引用 (Capture)** 了 `self.sigmas_old`。
当 JAX 对这个函数进行 `jit` 编译时，它会将 `self.sigmas_old` 的**数值**作为常数（或追踪的 Tracer）编译进计算图中（对于纯函数），或者更常见的，`jax_fem` 框架会在每一步求解前，通过 `problem.update_stress_strain` **显式地原地修改 (In-place mutation)** `problem` 对象里的数组内容。
注意：在 JAX 的 `scan` 循环或者 `while_loop` 中，这些变量通常需要作为 `carry` 显式传递，但在外部 Python `for` 循环（如本代码的第 215 行）中，我们可以利用 Python 对象的可变性来“偷懒”传递状态。

# FEM-JAX-GEO Project Presentation - Simplified Script

**Presenter**: Li Shiyao  
**Date**: January 2, 2026  
**Institution**: Hokkaido University  
**Total Duration**: About 25-30 minutes

---

## Slide 1: Title Slide

Good afternoon, Professor, and everyone.

My name is Li Shiyao.

Today, I want to present my research project, called **"FEM-JAX-GEO"**.

This project aims to build a differentiable finite element analysis framework for geomechanics. By using JAX and Automatic Differentiation, we're trying to solve some big problems in traditional inverse analysis—especially the difficulty of computing gradients for complex plastic models.

In this talk, I'll go through the research background, the technical framework I chose, my current progress on the Drucker-Prager model, and finally, my future plans.

---

## Slide 2: Outline

Let me start with an outline of today's talk.

There are four main parts.

First, the **Research Background**. I'll explain the challenges in geotechnical inverse analysis—what we call the "Curse of Dimensionality" and the "Gradient Problem."

Second, the **Technical Framework**. I'll explain why I chose JAX, and how the Drucker-Prager model fits into this design.

Third, my **Completed Work**. This is the main technical part. I'll show verification results of my code, proving that Automatic Differentiation works correctly with machine-level precision.

Finally, my **Future Plan**. I'll talk about the triaxial test simulations I plan to do and the roadmap for parameter inversion.

---

## Slide 3: Research Background - Inverse Analysis in Geomechanics

Let's start with the background.

In geotechnical engineering, getting accurate material parameters is always a big challenge. Lab tests are limited in scale and don't always show the real behavior of soil and rock in the field.

This is where **Inverse Analysis** helps. The basic idea is simple: we try to estimate material parameters—like Young's modulus, cohesion, or friction angle—by matching our simulation results with real field data, like displacement from sensors.

In math terms, this is an optimization problem. We want to find the parameters that minimize a loss function—basically, the difference between what our FEM model predicts and what we actually see in the field.

Here's the challenge: the forward simulation—running the FEM model with given parameters—is standard and well-known. But the "backward" step—updating parameters to reduce the error—is very hard in traditional tools.

---

## Slide 4: Challenges of Traditional Methods

So, what exactly are the problems? I've listed three main challenges.

The first is the **Curse of Dimensionality**. Traditional optimization methods like Genetic Algorithms or Particle Swarm don't need gradients. That sounds convenient, but the problem is: their cost grows very fast as the number of parameters increases. For complex problems with many parameters, these methods become too slow.

The second is the **Gradient Problem**. We know from optimization theory that gradient-based methods, like L-BFGS, are much faster. We want to use them. But traditional FEM software like Abaqus, Plaxis, or FLAC3D work as "black boxes." They can run forward simulations well, but they don't give us the gradients we need.

This leads to the third problem: the **Efficiency Bottleneck**. When we want to use gradient methods but can't get real gradients, the only choice is **Finite Difference**. This is very expensive. If we have N parameters, we need N+1 simulations just to get one gradient estimate. For problems with thousands of parameters, this is simply not possible.

---

## Slide 5: The Solution - Differentiable Physics Framework

To solve these problems, we propose a new approach: **Differentiable Physics** based on Automatic Differentiation, or AD.

Please look at the comparison on this slide.

On the left is the traditional Finite Difference way. Its cost grows with the number of parameters N. It's slow and has numerical errors.

On the right is our solution using JAX. By making the FEM solver "differentiable"—meaning every operation can be traced—we can compute gradients using **Backpropagation**, just like training a neural network.

The key benefit is the cost. With this approach, cost becomes **O(1)**, or constant. Whether we have one parameter or one million, the cost to compute gradients is about the same: one forward pass and one backward pass. Plus, JAX runs on **GPUs**, giving us huge parallelism.

---

## Slide 6: Core Technology - What is JAX?

Now, let me talk about the core engine: **JAX**, made by Google Research.

JAX is not just another deep learning library like TensorFlow or PyTorch. It's a system for **function transformations** built on NumPy.

It has four key features:

First, **`jax.grad`**: This lets us differentiate any Python code, including loops and if-else statements. This is important because our FEM solver has many conditionals in the plasticity code.

Second, **`jax.jit`**: This means Just-In-Time compilation. JAX uses XLA compiler to make our Python code run as fast as optimized C++.

Third, **`jax.vmap`**: This means automatic vectorization. It turns a function for one element into a function for many elements at once. This is perfect for FEM, where we apply the same model to every integration point.

Finally, JAX runs on **GPUs and TPUs**, giving us big parallelism without writing CUDA code.

---

## Slide 7: The JAX-FEM Library

We built our work on **JAX-FEM**, an open-source library by Takatani et al. in 2022.

As shown in the diagram, JAX-FEM sits on top of JAX and gives us the basic building blocks for finite element analysis.

It includes the `Mesh` module for geometry, the `FiniteElement` module for shape functions and integration, and the `Solver` module that uses solvers like PETSc and MUMPS for solving equations.

But the original JAX-FEM mainly focused on hyperelasticity and fluid problems. Our work extends it to handle **Geomechanics**—specifically, plasticity models that are essential for soil and rock behavior.

---

## Slide 8: FEM-JAX-GEO Project Architecture

This slide shows the overall structure of our **FEM-JAX-GEO** project.

The project has three main parts.

First, the **Infrastructure** layer. We use JAX and JAX-FEM as the base that provides FEM features.

Second, the **Physics** layer. This is where I spent most effort so far. It involves implementing the **Drucker-Prager plasticity model** and its return mapping algorithm in a fully differentiable way.

Third, the **Application** layer—our final goal. This involves building an **Inverse Analysis** module that uses gradients to optimize parameters efficiently.

Today, I'll focus mainly on the **Physics** part—the implementation details and verification, since this is the foundation for everything else.

---

## Slide 9: The Drucker-Prager Constitutive Model

Now, let's look at the physics model we chose.

We implemented the **Drucker-Prager model**, one of the most used plasticity models in geotechnical engineering.

Why this model? Unlike metal plasticity which uses Von Mises, soil and rock strength depends on **confining pressure**. Soil gets stronger when confined. The Drucker-Prager model captures this pressure-dependent behavior while being simpler than Mohr-Coulomb, which makes it easier to implement in a differentiable framework.

The yield function is shown in the equation. It depends on I1 (hydrostatic pressure) and J2 (shear stress).

Geometrically, as shown in the 3D plot, this represents a **cone** in stress space. The cone gets wider as pressure increases. This nicely captures the frictional nature of geomaterials—more pressure, more strength.

---

## Slide 10: Stress Return Mapping Algorithm

Implementing plasticity needs an iterative algorithm called the **Return Mapping Algorithm**. Let me explain how it works.

There are three main steps at every integration point during every load step.

**Step 1: Elastic Predictor.** We first assume the strain is purely elastic. We calculate a "trial stress" by adding elastic stress to the previous stress.

**Step 2: Yield Check.** We check the trial stress in the yield function f. If f is less than or equal to zero, the material is still elastic, and we're done.

**Step 3: Plastic Corrector.** But if f is greater than zero, the stress is outside the yield surface. We must "return" it to the surface by calculating the plastic multiplier and moving the stress back.

Implementing this in JAX is not easy, because we need gradients to flow correctly through all these conditions and nonlinear operations.

---

## Slide 11: Challenges in Differentiability

Implementing plasticity in a differentiable framework has some big challenges.

The main issue is the **Non-Smooth Nature** of plasticity. The change from elastic to plastic is a sharp "kink" in the stress-strain curve—it's not smooth. Also, the tip of the Drucker-Prager cone is a singularity where the gradient is undefined.

At these non-smooth points, automatic differentiation can fail or produce NaN values, which would break our optimization.

To handle these, we applied several techniques:

First, **Vertex Regularization**. We smooth the sharp tip of the cone by adding a small constant inside the square root. This rounds off the singularity.

Second, **Softplus Smoothing**. We use Softplus function—a smooth version of ReLU—to smooth the yield transition.

Third, **Viscoplastic regularization** using the Perzyna model, which adds a small rate-dependency that helps stabilize gradients.

---

## Slide 12: Completed Work - Code Structure

Now let's look at what we've actually implemented.

This slide shows the project code structure.

The main logic is in `src/models/drucker_prager.py`. This Python file, about 130 lines long, contains the complete differentiable Drucker-Prager plasticity model with all the return mapping logic.

The `examples` folder has simulation scripts for different test cases.

And the `tests` folder is very important—it has the scripts we use to verify the accuracy of our gradients, which we'll look at next.

---

## Slide 13: Core Achievement 1 - Constitutive Implementation

The first achievement is the `DruckerPragerPlasticity` class.

We built this by inheriting from the base `Problem` class from JAX-FEM. This gives us clean integration with the existing solver.

The most important function is `stress_return_map`. As you can see in the code, it contains all the physics: elastic prediction, yield check, and plastic correction.

This function is written in pure JAX using operations like `jnp.where` for conditionals. It's "stateless"—it doesn't use external mutable state—and works with both JIT compilation and automatic differentiation.

---

## Slide 14: Verification Experiment - Block Compression Setup

To verify our implementation, we did a standard **Block Compression Test**.

We set up a simple 10-millimeter cube with a 5×5×5 mesh, giving us 125 HEX8 elements.

For boundary conditions, we fixed the bottom and applied compression at the top, pushing the material into the plastic range.

The goal is a "Sanity Check": Does our FEM solver produce the same stress as a theoretical single-point calculation?

If they match, it proves that the global stiffness matrix assembly and Newton-Raphson solver work correctly.

---

## Slide 15: Verification Results - Perfect Match

Now let me show the results.

On the left is the stress-strain curve. The blue line is the theoretical reference. The red dots are from our JAX-FEM solver.

They match perfectly. The red dots sit exactly on the blue line. The error is **0.00%**, within numerical precision.

On the right is the 3D stress field in ParaView. The stress is uniform throughout the cube, as expected for this simple loading.

This proves that the **forward simulation** is completely correct.

---

## Slide 16: Core Achievement 3 - Gradient Verification

Now the most important part: Is the code truly differentiable?

To verify this, we compared gradients from **JAX AD** against **Finite Difference** as our reference.

As shown in the table, JAX's AD uses the exact chain rule. This should give machine precision—about 10^(-16).

Finite Difference is an approximation with typical error around 10^(-6).

We used Finite Difference as a "ground truth" to check that our JAX implementation gives correct gradients.

---

## Slide 17: Detailed Gradient Experiment Results

This table shows our gradient verification results.

We tested gradients of the loss function with respect to Young's Modulus at different loading levels.

The test cases included: small elastic strain, initial yielding, and large plastic deformation.

The results are excellent. The error between AD and FD is in the range of **10^(-13) to 10^(-16)**.

This means we achieve **machine-level precision**. It proves the differentiation handles the complex conditional logic correctly—all the if-else branches, square roots, and regularization.

This shows we can now use these exact gradients for parameter optimization with very high accuracy.

---

## Slide 18: Critical Challenge - Boundary Conditions

Now I want to share a critical issue we found, because it gives important insights.

At first, our gradient verification completely failed. We had gradient errors of hundreds or thousands of percent.

After a lot of debugging, we found the cause: **Rigid Body Modes**—a classic issue in structural mechanics.

In our first setup, we only fixed vertical displacement at the bottom. But the cube was still free to slide horizontally and rotate.

In math terms, this made the stiffness matrix K **singular**—its determinant is zero.

Here's the interesting part: The forward solver could still work because PETSc can handle singular matrices by picking one solution from many possible ones.

But the **backward pass**—the Adjoint Equation for gradients—failed completely. When K is singular, the adjoint equation requires special conditions (the Fredholm Alternative Theorem) to have a solution.

Our setup violated these conditions, so the adjoint had no solution, and gradients exploded.

By fully fixing the bottom boundary—setting u_x, u_y, and u_z all to zero—we removed the rigid body modes. Immediately, gradient error dropped to machine precision.

---

## Slide 19: Short-Term Plan - Triaxial Simulation

Now let's talk about future plans.

In the short term—the next 1-2 months—my main goal is to simulate a standard **Triaxial Compression Test**. This is the most common experiment in geotechnical engineering for testing soil behavior.

Unlike the block compression test, triaxial tests require **Confining Pressure** using force boundary conditions. Then we apply axial loading to failure.

I plan to implement complex loading paths to match the stress paths in real lab triaxial tests. This will be the foundation for validating against real data—a critical step before inverse analysis.

---

## Slide 20: Mid-Term Plan - Parameter Inversion

In the mid-term—about 3-6 months—I'll focus on the core **Parameter Inversion** framework, which is our ultimate goal.

I'll implement the complete optimization loop:

**Step 1**: Run Forward FEM with current parameter estimates.

**Step 2**: Calculate the Loss between simulation and observed data.

**Step 3**: Use `jax.grad` to compute exact gradients—this is where our differentiable FEM shines.

**Step 4**: Update parameters using optimizers like Adam or L-BFGS, which can use these exact gradients to converge much faster.

A key challenge is handling **Spatially Variable** parameters. Real soils vary in space. I plan to use Random Field methods to model this and invert the whole spatial field, which could have thousands of parameters. This is where JAX's efficiency really matters.

---

## Slide 21: Summary of Achievements

Let me summarize what we've achieved.

First, I successfully implemented the **Drucker-Prager** plasticity model in JAX, creating a fully differentiable model for geomaterials.

Second, I verified forward simulation accuracy with **0.00% error** against theoretical benchmarks.

Third, I proved that **Automatic Differentiation** works perfectly with plasticity, achieving gradient errors down to 10^(-16) across elastic, yielding, and fully plastic states.

Fourth, I found and solved a critical stability issue with rigid body modes, which gives important lessons for future work.

These achievements lay a solid foundation for the next phase—real inverse analysis.

---

## Slide 22: Current Challenges

Despite progress, several challenges remain.

First, **Gradient Stability**. Our implementation works well for simple compression, but ensuring stable gradients under complex loading—especially cyclic loading—is still difficult.

Second, **Computational Efficiency**. While JAX and GPU help a lot, we need to further optimize for very large meshes with millions of elements. Memory management and reducing CPU-GPU data transfer will be important.

Third, **Experimental Validation**. To show real value to the geotechnical community, we need real triaxial test data to benchmark our inverse analysis. We need to show we can recover meaningful parameters from real data, not just synthetic tests.

---

## Slide 23: Thank You

This concludes my presentation on FEM-JAX-GEO.

I want to thank Professor and the whole research group for your support, guidance, and feedback.

I'm happy to take any questions about the technical details, implementation, or future directions.

Thank you very much for your attention.

---

## Appendix Slide A3: Rigid Body Modes & Adjoint Equation

Let me briefly explain the math behind the boundary condition issue.

On the left is the forward equation: K × u = F. Without full constraints, K has a null space—there exist rigid body mode vectors v_rbm where K × v_rbm = 0. This means K is singular.

On the right is the **Adjoint Equation** for gradients: K^T × λ = -dL/du, where λ are adjoint variables and the right side is g (gradient of loss with respect to displacement).

According to the **Fredholm Alternative Theorem**, for this equation to have a solution when K is singular, g must be orthogonal to the null space of K^T. The dot product of g with every rigid body mode must be zero.

In our first setup, this condition was not met. The gradient g had components in rigid body mode directions, so the dot product was not zero. Therefore, the adjoint equation had **no solution**, and gradients blew up.

By fixing u_x, u_y, and u_z at the bottom, we removed the null space, made K non-singular, and the adjoint equation became solvable.

---

## Appendix Slide A4: Experimental Environment

Finally, let me give details about our experimental setup.

For hardware, we use an **NVIDIA A100 GPU** with **CUDA 12**. The A100 has 40GB of high-bandwidth memory. This hardware is key for JAX's JIT compilation and for parallelizing FEM element calculations.

For software, the project uses **Python 3.10+**. Besides JAX for automatic differentiation and compilation, we use:

**PETSc**—Portable, Extensible Toolkit for Scientific Computation—for linear algebra and solvers.

**MUMPS**—MUltifrontal Massively Parallel Sparse direct Solver—as the direct solver in PETSc for the sparse linear systems in Newton-Raphson iterations.

This combination of modern GPU hardware and carefully chosen software lets us achieve both the accuracy for scientific computing and the performance for practical inverse analysis.

---

## End of Presentation

(End of simplified script)

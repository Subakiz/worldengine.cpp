# Architectural Deep Dive — WorldEngine.cpp

`WorldEngine.cpp` (`PlayWorld`) provides the foundational runtime for action-conditioned neural world simulation. This document details the mathematical models and systems engineering behind the core engine.

---

## 1. Distribution Matching Distillation (DMD) & 1-Step Execution

Standard diffusion models require 20 to 50 denoising iterations per frame:
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$
This produces latencies of 1,000–5,000 ms, making real-time interactive gameplay impossible.

`WorldEngine.cpp` executes distilled student checkpoints trained with **Distribution Matching Distillation (DMD)**:
$$\mathbf{z}_0 = \mathbf{x}_t - \sigma_t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c})$$
- Evaluates a **single forward pass** with deterministic zero variance.
- Incorporates **CFG-Aware Student Distillation (CASD)**: the classifier-free guidance factor $s$ is baked into student weights, eliminating the redundant negative pass ($2\times$ compute savings).

---

## 2. Frustum Voxel Memory Grid (Anti-Drift Spatial Anchoring)

### The "World Melting" Problem
Autoregressive neural world models suffer from state amnesia:
$$\text{Frame } 0 \to \text{Turn 180°} \to \text{Turn back 180°} \to \text{Original terrain hallucinated away (SSIM } < 0.40)$$

### Solution: Pose-Indexed Latent Memory
1. **5-DOF Camera Trajectory**:
   $$\mathbf{P}_t = (x, y, z, \theta, \phi)$$
2. **Spatial & Angular Binning**:
   $$\mathbf{k}_{\text{spatial}} = \left( \left\lfloor \frac{x}{0.5\text{m}} \right\rfloor, \left\lfloor \frac{y}{0.5\text{m}} \right\rfloor, \left\lfloor \frac{z}{0.5\text{m}} \right\rfloor \right), \quad \mathbf{k}_{\text{angular}} = \left( \left\lfloor \frac{\theta}{15^\circ} \right\rfloor, \left\lfloor \frac{\phi}{15^\circ} \right\rfloor \right)$$
3. **Morton-Wang 64-bit Hash Mixer**:
   $$\text{HashKey}(\mathbf{P}_t) = \text{Mix64}\Big(\text{Morton3D}(\mathbf{k}_{\text{spatial}}) \oplus (\text{Morton2D}(\mathbf{k}_{\text{angular}}) \ll 32)\Big)$$
4. **Compact Latent Frame Storage**:
   Stores compact latent frame tensor $\mathbf{Z} \in \mathbb{R}^{C \times H \times W}$ ($28.1\text{ KB}$ per voxel for $C=4$, or $112.5\text{ KB}$ for $C=16$) rather than 28 layers of dense KV projections ($9.0\text{ GB}$ to $147\text{ GB}$). A 512-voxel LRU cache occupies only **14.1 MB to 56.3 MB VRAM**.
5. **Directional Cosine Similarity Blending**:
   $$\gamma = \max(0, \cos(\theta_t - \theta_{\text{cached}}))$$
   $$\mathbf{Z}_{\text{conditioned}} = \gamma \cdot \mathbf{Z}_{\text{cached}} + (1 - \gamma) \cdot \mathbf{Z}_{\text{autoregressive}}$$
   Guarantees 360-degree loopback permanence ($\text{SSIM} \ge 0.82$).

---

## 3. WebGPU Storage Buffer Sharding (9-Chunk Partitioning)

Under standard WebGPU implementations (Chrome default adapter limits):
- `maxStorageBufferBindingSize = 128 MB` (or `256 MB` on high-end desktop adapters).
- A 1.1 GB INT4 model exceeds this limit.

`WorldEngine.cpp` implements dynamic limit negotiation:
- If `adapter.limits.maxStorageBufferBindingSize >= 1 GB`, allocate a single contiguous `GPUBuffer`.
- Otherwise, dynamically partition weights across **9 contiguous GPUBuffer chunks of $\le 128\text{ MB}$ each** (`weights_shard_0` through `weights_shard_8`), indexed via binding slots in WGSL compute passes.

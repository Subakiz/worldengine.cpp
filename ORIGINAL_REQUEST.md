# Original User Request

## 2026-09-04T00:44:52Z

Evaluate high-impact open-source software project concepts that do not currently exist on GitHub to identify the single project with the highest probability of achieving massive GitHub star adoption and virality. Produce a concise comparative evaluation report followed by a concrete, actionable implementation plan for the winning project.

Working directory: /Users/nabils/antigravity/open source
Integrity mode: development

## Requirements

### R1. Comparative Opportunity Report
Evaluate candidate software projects that lack a mature or complete open-source equivalent on GitHub. Benchmark each candidate across quantifiable dimensions: latent developer demand, viral appeal (social media / Hacker News hook), star velocity of comparable open-source successes, technical feasibility, and defensibility. Conclude with an unambiguous selection of the #1 project best positioned to maximize GitHub stars.

### R2. Winning Project Implementation Plan
Produce an actionable technical implementation plan for the selected winner. Specify the target architecture, modular component structure, dependencies, initial MVP feature scope, and a concrete phased development roadmap.

### R3. GitHub Launch & Virality Strategy
Define the go-to-market playbook tailored specifically for GitHub star velocity: repository positioning, hero README structure with killer demonstration hooks, documentation strategy, and community distribution channels.

## Acceptance Criteria

### Comparative Report Validation
- [ ] Compares at least 4 distinct project categories (e.g. AI developer tooling, world models/frontier AI reproduction, local-first infra, systems virtualization).
- [ ] Contains a structured scoring matrix with explicit numerical or categorical rubrics for: viral hook, market demand, competition gap, and implementation complexity.
- [ ] Formally crowns a single winning project with documented rationale on why it outperforms alternatives in star-gathering potential.

### Technical Implementation Plan Validation
- [ ] Breaks down the winning system into discrete components with clear interface boundaries.
- [ ] Details an MVP milestone reachable within a rapid prototyping cycle that yields a functional, demonstrable core.
- [ ] Defines verifiable acceptance tests and quality benchmarks for the MVP.

### Launch & Star Acceleration Strategy Validation
- [ ] Outlines the exact structure and visual hook of the top-of-README presentation (e.g., GIF/video demo concept, interactive quickstart).
- [ ] Identifies target developer ecosystems and distribution vectors (Hacker News Show HN strategy, subreddit alignment, Twitter/X demo angles).

## 2026-09-04T01:06:30Z

Plan, implement, thoroughly test, and prepare for GitHub release the end-to-end MVP of `WorldEngine.cpp` (`PlayWorld`): the zero-dependency C++20 and WebGPU runtime for interactive neural world models based on `WINNING_PROJECT_PLAN.md`.

Working directory: /Users/nabils/antigravity/open source
Integrity mode: development

## Requirements

### R1. Core C++20 Engine Subsystems
Implement the core `WorldEngine.cpp` modular architecture conforming to `WINNING_PROJECT_PLAN.md`:
- **Model Container & Quantization**: `.PWMF` (PlayWorld Model Format) binary parser and writer supporting INT4/FP8/FP16 block tensor storage with header validation and memory-mapped loading.
- **Action Condition & Input Bridge**: Lock-free `ActionRingBuffer` capturing normalized continuous mouse deltas $(\Delta x, \Delta y)$ and discrete key bitfields (WASD, Space) mapped into action conditioning vectors.
- **Frustum Voxel Memory Grid**: Spatial pose-indexed latent anchor cache to prevent world amnesia and drift during camera rotation.
- **Inference Pipeline**: Step-wise causal temporal scheduler coordinating state transitions.

### R2. Interactive Browser WebGPU Player & Native Benchmark Harness
- **WebGPU HTML5 Interactive Player**: Self-contained web client with HTML5 Canvas, Pointer Lock, and WebGPU WGSL compute shaders (or Emscripten Wasm bridge) that runs an interactive simulation loop at interactive frame rates with WASD + mouse steering.
- **Desktop Benchmark Executable**: Standalone native CLI executable (`worldengine-bench`) that loads a `.PWMF` model, executes $N$ simulated action steps, and measures microsecond-level per-frame latency, memory footprint, and throughput.

### R3. Automated Test Suite & Quality Verification
- Implement a comprehensive test suite (unit tests and integration tests) covering tensor operations, `.PWMF` serialization roundtrips, ring-buffer concurrency, voxel grid spatial indexing, and action conditioning.
- Provide a unified test runner script that executes all tests and outputs structured verification metrics.

### R4. Git Repository Initialization, CI Workflows & GitHub Release Packaging
- Initialize a local Git repository with clean, atomic commits following Conventional Commits.
- Create GitHub Actions CI workflows (`.github/workflows/ci.yml`) for multi-platform build and test verification.
- Author the production-grade `README.md` implementing the virality hooks, demo showcase layout, architecture breakdown, and quickstart guide specified in `GITHUB_LAUNCH_STRATEGY.md`.
- Create a tagged initial release (`v0.1.0-alpha`) ready for pushing to GitHub.

## Acceptance Criteria

### Core Engine & Formats
- [ ] Code compiles cleanly with C++20 (Clang/GCC) with zero fatal compilation errors or missing symbols.
- [ ] `.PWMF` reader/writer can successfully serialize and deserialize model layers with byte-level checksum verification.
- [ ] `ActionRingBuffer` handles rapid concurrent push/pop without race conditions or data corruption.
- [ ] Frustum Voxel Memory correctly stores and retrieves spatial latent anchors based on camera pose hashing.

### Player & Benchmark
- [ ] Standalone benchmark CLI executes 100 consecutive forward steps and outputs average latency per step and memory consumption.
- [ ] Web client directory contains all necessary HTML/JS/WGSL assets and can be served locally via a simple HTTP server (`python3 -m http.server`).

### Verification & Testing
- [ ] Automated test suite runs with 100% pass rate across all unit and integration tests.
- [ ] Memory leaks and invalid accesses are verified clean via sanitizers or memory audit tests.

### Repository & Release Readiness
- [ ] Git repository is initialized with a structured commit history and no uncommitted scratch files.
- [ ] GitHub Actions CI workflow file is syntactically valid and tests the build pipeline.
- [ ] Production `README.md` includes interactive quickstart instructions, architecture diagram, and feature roadmap.
- [ ] Tag `v0.1.0-alpha` exists on the main branch.

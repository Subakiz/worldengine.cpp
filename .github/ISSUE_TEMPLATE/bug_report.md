---
name: "Bug Report"
about: "Report a reproducible defect, crash, or unexpected behavior in WorldEngine.cpp"
title: "[BUG] "
labels: ["bug", "triage"]
assignees: ""
---

## Summary
<!-- Provide a clear and concise description of the bug. -->

## Steps to Reproduce
1. Command executed (e.g., `./build/bin/worldengine-bench --model ...`):
2. Model used (`.pwmf` filename, precision INT4/FP8, origin):
3. Configuration flags / environment variables:
4. Steps taken / input actions:

## Expected Behavior
<!-- A clear description of what you expected to happen. -->

## Actual Behavior
<!-- What actually occurred (e.g., crash, assertion failure, visual artifact, memory spike). -->

## System Environment & Hardware
- **Operating System & Version**: [e.g., macOS Sonoma 14.6 arm64, Ubuntu 24.04 x86_64, Windows 11]
- **CPU**: [e.g., Apple M3 Max, AMD Ryzen 7 7800X3D, Intel Core i9-14900K]
- **GPU & VRAM**: [e.g., Apple M3 Max 38-core, NVIDIA RTX 4090 24GB, RTX 3060 12GB]
- **Compiler & Version**: [e.g., Apple Clang 15.0.0, GCC 13.2.0, MSVC 19.38]
- **Compute Backend**: [e.g., Apple Metal, WebGPU (WGSL), NVIDIA CUDA, CPU Reference]
- **Browser (if WebGPU player)**: [e.g., Google Chrome 128.0, Microsoft Edge 128.0, Safari 18.0]

## Test Runner & Sanitizer Output
<!-- Please run the automated test suite and paste the output: -->
```text
$ ./scripts/run_all_tests.sh
# Output:

$ ./scripts/run_all_tests.sh --sanitize
# Sanitizer (ASan / UBSan) output if applicable:
```

## Crash Logs / LLDB / GDB Backtrace
<!-- If a crash occurred, paste the stack backtrace below: -->
```text
(lldb) bt
# or
(gdb) bt full
```

## Additional Context
<!-- Add any other context or screenshots about the problem here. -->

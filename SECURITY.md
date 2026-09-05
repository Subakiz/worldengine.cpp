# Security Policy & Vulnerability Disclosure

The `WorldEngine.cpp` (`PlayWorld`) team takes the security and stability of our high-performance runtime seriously. Because `WorldEngine.cpp` parses binary model files (`.PWMF`), performs SIMD-accelerated math on raw memory buffers, and orchestrates WebGPU shaders, we maintain a defensive systems programming posture.

---

## 1. Supported Versions

We provide active security updates, vulnerability remediation, and backports for the following releases:

| Version | Supported | Security Maintenance Status |
| :--- | :---: | :--- |
| `0.1.x` | **Yes** | Active security patches and critical vulnerability backports |
| `< 0.1.0` | **No** | Alpha / prototype releases; please upgrade to `v0.1.0` or newer |

---

## 2. Attack Surfaces & Scope

### 2.1 In-Scope Attack Surfaces
We actively investigate and patch vulnerabilities in:
- **Binary Model Container (`.PWMF`) Parser**:
  - Buffer overflows, out-of-bounds reads or writes during header/tensor parsing.
  - Integer overflow or wraparound in chunk offset/length calculations.
  - Validation bypasses for IEEE 802.3 CRC32 checksums.
  - Malformed tensor shapes, unhandled data type enums, or non-finite float exploitation (NaN/Inf injections).
- **Concurrent Shared Memory & Lock-Free Queues**:
  - Torn reads/writes, race conditions, or memory corruption in `ActionRingBuffer`.
  - Memory reordering bugs violating acquire-release invariants.
- **Dequantization & SIMD Math Kernels**:
  - 64-byte alignment violations triggering CPU hardware faults.
  - Memory safety violations in INT4/FP8/FP16 dequantization kernels.
- **Browser WebGPU Client**:
  - Storage buffer overruns in WGSL compute shaders.
  - Memory exhaustion vulnerabilities in the browser runtime.

### 2.2 Out-of-Scope Surfaces
The following items are outside the scope of this security policy:
- Vulnerabilities in underlying third-party GPU hardware, operating system kernels, or browser WebGPU implementations (please report these to Chromium/WebKit/GPU vendor teams).
- Denial-of-service attacks against third-party self-hosted demo servers.
- Generative hallucinations, visual artifacts, or non-factual generation produced by neural diffusion models (these are inherent properties of generative AI, not software vulnerabilities).

---

## 3. Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or social media.**

### Primary Method: GitHub Private Vulnerability Reporting (Recommended)
1. Navigate to the repository's **Security** tab: `https://github.com/playworld/worldengine.cpp/security`
2. Click **Report a vulnerability**.
3. Provide a detailed summary, proof-of-concept (PoC) model file or script, and affected version.

### Secondary Method: Encrypted Email
If you are unable to use GitHub Security Advisories, email:
**security@playworld.run**

Please include:
- A description of the vulnerability and its potential impact.
- Exact reproduction steps, including any malformed `.pwmf` files or command invocations.
- System environment (OS, CPU/GPU, compiler version, build flags).
- Sanitizer traces (AddressSanitizer / UndefinedBehaviorSanitizer output), if available.

---

## 4. Response Timeline SLA (Service Level Agreement)

We adhere to a strict response timeline for all reported vulnerabilities:

| Stage | Response SLA Window | Description |
| :--- | :---: | :--- |
| **Initial Acknowledgment** | **Within 24 hours** | Maintainer confirms receipt of report and opens private communication channel. |
| **Triage & Severity Assessment** | **Within 48 hours** | Vulnerability is reproduced, assigned CVSS score, and classified. |
| **Status Updates** | **Every 72 hours** | Ongoing progress updates provided to the reporter until resolution. |
| **Patch & Coordinated Advisory** | **Within 14 calendar days** | Release patch tagged, validated across sanitizers, and published with CVE/GHSA. |

---

## 5. Coordinated Disclosure & Hall of Fame

We follow responsible disclosure principles. We request that reporters maintain confidentiality until an official patch and advisory are published.

Upon resolution, security researchers who responsibly report verified vulnerabilities will be credited in:
- The GitHub Security Advisory release notes.
- The `RELEASE_NOTES.md` changelog.
- Our community Security Hall of Fame.

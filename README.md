# Rineto

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/PyO3-0.25-blue.svg)](https://pyo3.rs/)
[![Status](https://img.shields.io/badge/Status-Archived%20Artifact-lightgrey.svg)]()

---

## ⚠️ Status: Archived Research Artifact

**This repository contains old, experimental files recovered during a computer cleanup and published as-is.**

The code was written approximately a year ago as a student research project exploring neuro-symbolic AI architectures, custom quantization formats, and cognitive-inspired memory systems. It was never brought to production quality and was recently rediscovered while cleaning up the development machine.

**What this repository is:**
- A collection of Rust source files (`.rs`) exploring various AI architecture concepts.
- A conceptual reference for researchers interested in alternative memory structures, quantization schemes, or neuro-symbolic orchestration.
- Raw experimental code, not a maintained library.

**What this repository is NOT:**
- A production-ready library.
- A maintained project with active development.
- Code with tests, benchmarks, or CI/CD.
- A complete crate (no `Cargo.toml` is included; dependencies must be reconstructed).

If any concept or snippet proves useful to your work, feel free to use it under the MIT License. No guarantees of correctness or compatibility with current toolchains are provided.

---

## 📦 Repository Contents

This repository contains **only Rust source files** (`.rs`). No configuration files, build scripts, examples, or documentation beyond this README are included.

### File Inventory

| File | Description |
|---|---|
| `lib.rs` | Core module: graph structures, ASM DSL, routing pipeline, hash functions, BPE tokenizer, MoE, optimizer, Transformer blocks, Attention, FFN, RMSNorm, SSM, vision primitives, cognitive layers (`RinetoMind`, `RinetoInnerWorld`, `RinetoExperience`), diffusion pipeline, scheduler, metrics, PyO3 module exports. |
| `quantized.rs` | Custom quantization formats: `RinetoBR1`, `RinetoIR1`, `RinetoBR4`, `RinetoIR4`, `RinetoBR16`. Includes AVX2-optimized matmul for 16-bit format. |
| `rinetofloat.rs` | Custom `RinetoFloat16` format and `RinetoMatrixF16` for low-precision floating-point representation. |
| `tensorrineto.rs` | `TensorRineto`: tensor primitive with u16 quantization, spherical normalization, matmul with caching, RMSNorm, SiLU activation. |
| `rthink.rs` | `RThink`: declarative six-point chain-of-thought scratchpad (`цель → план → мысль → сомнение → проверка → рефлексия`) with per-kind temperature sampling. |
| `sferom.rs` | `RinetoSferom`: hierarchical spherical associative memory (CORE → LOBE → nanos) with triple-score routing (cosine + Hamming + Jaccard). |
| `engram.rs` | `RinetoEngram`: hash-table associative memory mapping text to float vectors. |
| `cascade.rs` | `RinetoCascade`: domain-aware model routing (chat vs. think models) based on query classification. |
| `transplant.rs` | `RinetoTransplant`: weight transplantation from external LLM formats into `RinetoLM`, with optional quantization to BR4/IR4 formats. |

---

## 🧠 Core Concepts

### 1. SFEROM — Hierarchical Spherical Associative Memory

A three-level memory structure with triple-score routing:

score = 0.70 × cosine(query, centroid)
+ 0.20 × (1 - hamming_distance/64)
+ 0.10 × jaccard(query, nanos)


Centroids are updated via EMA and stay normalized on the unit sphere. Potential use cases: online RAG without re-indexing, MoE routing, long-context memory without external vector databases.

### 2. RThink — Declarative Chain-of-Thought Scratchpad

A six-point structured reasoning protocol:

цель → план → мысль → сомнение → проверка → рефлексия


Unlike ad-hoc CoT prompts, `RThink` is a typed, inspectable object that can be serialized, cached, and verified. Each reasoning kind has its own temperature parameter (e.g., 0.3 for logical, 1.2 for creative).

### 3. RinetoCascade — Domain-Aware Model Routing

A chat-vs-think cascade that routes CODE / MATH / LOGIC queries to a deeper reasoning model and everything else to a fast chat model. Analogous to FrugalGPT but implemented at the inference-engine level.

### 4. Custom Quantization Formats

Five custom formats with "Rineto formula":
- `BR1` / `IR1` — 1-bit binary/integer
- `BR4` / `IR4` — 4-bit balanced/integer
- `BR16` — 16-bit balanced with AVX2-optimized matmul

All formats include forward (quantize) and backward (dequantize) paths. Note: current `matmul` implementations dequantize to `f32` during computation, so memory savings apply primarily to storage, not peak RAM usage.

### 5. RinetoASM — Declarative Orchestration DSL

A safe assembly-like language for pipeline routing with static validation:

CLASSIFY_DOMAIN CODE
SELECT_EXPERT Coder
SELECT_TEMPLATE FixError
CALL_TOOL compiler
CALL_TOOL tests
VERIFY
RETURN

Useful as a controlled substrate for agent frameworks where arbitrary code execution is a security risk.
6. Rineto-hash — Custom Bitwise Hashing
An alternative to FNV-1a for SimHash-based semantic signatures: XOR-fold with rotation and golden-ratio multiplication. Produces locality-preserving 64-bit signatures without external dependencies.
7. Cognitive Layers
Experimental cognitive-inspired components:
RinetoMind — predictive processing with inner speech and metacognition
RinetoInnerWorld — autobiographical memory, self-concept, mirror test
RinetoExperience — unified cycle of emotion, prediction, surprise, and self-development
RinetoEmotions — emotion computation from internal state vectors
These are functional models, not claims of consciousness. They simulate behavioral signs of cognition without asserting subjective experience.

┌─────────────────────────────────────────────────────────────┐
│                      RinetoPipeline                          │
├─────────────────────────────────────────────────────────────┤
│  Query ──→ RinetoRouter ──→ RinetoASM ──→ Z3Verifier       │
│                              ↓                               │
│                    RouteDecision (domain, expert, template)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                            │
├─────────────────────────────────────────────────────────────┤
│  RinetoCascade ──→ RinetoLM (Transformer + SSM + RoPE)     │
│       ↓                                                      │
│  RinetoSferom (Memory) ←→ RinetoEngram (Hash Table)        │
│       ↓                                                      │
│  RThink (CoT Scratchpad) ──→ RinetoGraph (Symbolic Layer)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Low-Level Primitives                       │
├─────────────────────────────────────────────────────────────┤
│  TensorRineto │ Quantized Formats (BR1/IR1/BR4/IR4/BR16)   │
│  RinetoBPE    │ RinetoHash (SimHash)    │ AVX2 kernels      │
│  RinetoMMapWeights (mmap I/O)          │ XorShift64 RNG     │
└─────────────────────────────────────────────────────────────┘

Reconstruction Notes

This repository contains only .rs source files. To rebuild as a crate, the following approximate Cargo.toml structure was used during original development:

[package]
name = "rineto"
version = "0.1.0"
edition = "2021"

[lib]
name = "rineto"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
numpy = "0.25"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
memmap2 = "0.9"
rayon = "1.8"
libc = "0.2"

[profile.release]
opt-level = 3
lto = true

Warning: The exact dependency versions used originally were pyo3 = "0.25" and numpy = "0.25". API compatibility with current versions of these crates is not guaranteed. The code may require adjustments to compile against modern PyO3 (which has undergone significant API changes between 0.20 and 0.25+).

Known Issues and Limitations

This codebase was written by a student learning Rust and ML engineering simultaneously. The following issues are known and were not resolved before the project was archived:
Critical Issues
Double residual connection in RinetoFFN::forward — the FFN layer adds residual internally, but RinetoTransformerBlock::forward adds it again, resulting in y = 2x + FFN(x) instead of the standard y = x + FFN(x).
GQA implementation bug in RinetoAttention — when num_kv_heads < num_heads, weight indexing may access out-of-bounds memory due to incorrect size calculation.
transplant.rs FFN path mismatch — load_layer_bytes_quantized clears ffn.w1 / ffn.w2 after quantization, but ffn_silu_core still reads from the cleared fields. Quantized path needs explicit integration.
RinetoFloat16::from_f32 potential recursion — the PyO3 wrapper may call itself recursively depending on module resolution.
Moderate Issues
Constants with trailing spaces — many constants like DOMAIN_CODE = "CODE " have trailing spaces, causing comparison mismatches.
RinetoIR1::dequantize and RinetoIR4::dequantize return padded arrays instead of truncating to original length.
TensorRineto::add checks length equality but not shape equality, allowing [2, 3] + [3, 2] operations.
No KV-cache in generation — RinetoLM::generate recomputes full forward pass for each new token, resulting in O(n²) complexity.
Monolithic lib.rs — ~10,000+ lines in a single file; should be split into independent crates.
No unit tests or benchmarks — the code has not been validated against standard baselines (FAISS, vLLM, bitsandbytes).

For Researchers

This repository is shared as a conceptual research artifact. If you work on:
RAG systems → SFEROM offers an alternative to FAISS/Chroma with online learning
MoE routing → Triple-score routing (cosine + Hamming + Jaccard) may inspire new gating mechanisms
Neuro-symbolic AI → RinetoASM + RinetoGraph provide a declarative substrate
Quantization → Custom 1/4/16-bit formats (though not yet kernel-optimized beyond BR16)
Cognitive architectures → RinetoMind / RinetoInnerWorld / RinetoExperience explore functional models of self-awareness
State-space models → RinetoSSM implements Mamba-style selective SSM with ZOH discretization
If any concept proves useful in your work, even in modified form, a brief note to the contact below would be appreciated as feedback.

Contact

Vlad — Student, programming languages and AI architecture research

Email: kerot9ksin@gmail.com
Discord: kero222
GitHub: KeroT9Ksin

If any concept from this archived codebase proves useful to your research or product, a brief acknowledgment would be meaningful to the original author. If nothing fits, no reply is necessary — the opportunity to have the code read is already sufficient.

# Embedded ML Inference Optimization Patterns & Kernels

[![Gumroad](https://img.shields.io/badge/Full_Field_Guide-Touvettech-blue.svg)](https://touvettech.gumroad.com/l/edgeml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/Language-C++17-orange.svg)](#)

A minimal, zero-dependency C++17 reference implementation for low-latency integer neural network inference on memory-constrained embedded platforms (ARM Cortex-M, RISC-V, ESP32).

---

## Overview

Deploying deep neural networks to constrained embedded microcontrollers requires careful management of arithmetic precision, cache lines, and memory bandwidth.

This repository demonstrates:
- **INT8 fixed-point vector dot products** with 32-bit accumulation and four-way superscalar loop unrolling.
- **Fixed-point arithmetic requantization** using integer multiplier-and-shift operations (avoiding floating-point emulation).
- **Zero dynamic allocations** (`malloc`/`free` safe) for deterministic execution and real-time operating systems (RTOS).

---

## Quick Start

The core header is self-contained and header-only.

```cpp
#include "edge_ml_kernels.hpp"
#include <iostream>
#include <vector>

int main() {
    // Example: 8-element quantized activations & weights
    int8_t a[8] = {12, -45, 67, 10, -5, 24, 88, -12};
    int8_t b[8] = {5, 2, -18, 44, 90, -3, 11, -50};

    // Fast unrolled INT8 dot product -> INT32 accumulator
    int32_t acc = EdgeML::dot_product_int8(a, b, 8);

    // Requantize accumulator back to INT8 (multiplier = 1073741824, shift = 31, zero_point = 0)
    int8_t output = EdgeML::requantize(acc, 1073741824, 31, 0);

    std::cout << "Accumulator: " << acc << "\n";
    std::cout << "Requantized INT8: " << static_cast<int>(output) << "\n";
    return 0;
}
```

---

## Deep Dive: The Edge ML & Embedded Inference Field Guide

For full architectural blueprints, complete mathematical derivations, and memory hierarchy optimization on embedded SoCs, check out the complete **Edge ML Field Guide**:

👉 **[Get the Full Field Guide on Touvettech](https://touvettech.gumroad.com/l/edgeml)**

### Complete Guide Topics:
1. **Numerical Quantization Mechanics:**
   - Asymmetric vs. Symmetric affine quantization trade-offs.
   - Per-channel vs. Per-tensor scaling dynamics.
   - Quantization-Aware Training (QAT) vs. Post-Training Quantization (PTQ) error bounds.

2. **Memory Hierarchy & Bandwidth:**
   - The MCU Roofline model: identifying compute vs. memory-bound layers.
   - Ping-Pong double-buffering with direct memory access (DMA).
   - NCHW vs. NHWC tensor memory strides for SIMD cache lines.

3. **Graph Optimization & Engine Tuning:**
   - Operator fusion (Conv + BatchNorm + ReLU).
   - Verification SNR metrics between FP32 reference models and INT8 deployment engines.

---

## License

The code snippets in this repository are released under the [MIT License](LICENSE).

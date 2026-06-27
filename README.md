# TPU on FPGA — Complete Learning & Implementation Guide

> A structured roadmap for building a 4×4 Weight-Stationary Systolic Array TPU on FPGA,
> with application guides for a 0–9 Digit Classifier and Signal Classifier.

---

## Part 1 — Prerequisites: What to Learn Before You Start

### 1.1 Digital Logic Foundations

These are non-negotiable before writing a single line of Verilog.

| Topic | What to Focus On |
|-------|-----------------|
| Combinational logic | Gates, MUX, adders, comparators |
| Sequential logic | D flip-flops, registers, shift registers |
| Synchronous design | Clock domain, setup/hold time, why everything is edge-triggered |
| FSM design | Moore vs Mealy, state encoding, next-state logic |
| Arithmetic | Two's complement, overflow, signed vs unsigned operations |

**Why it matters for TPU:** Every PE is a register + multiplier + adder. The FSM drives your entire pipeline.

---

### 1.2 Verilog HDL

You need to be comfortable with synthesizable Verilog — not just simulation.

```
Topics to cover:
  - module, input, output, wire, reg
  - always @(posedge clk) blocks
  - Blocking (=) vs non-blocking (<=) assignments  ← most common bug source
  - Parameters and generate blocks
  - Signed arithmetic: reg signed [7:0]
  - Memory arrays: reg [7:0] mem [0:15]
  - Testbench writing: $readmemh, $monitor, $finish
```

**Key rule to internalize:** Use `<=` (non-blocking) inside clocked always blocks. Always. No exceptions.

**Practice exercises before starting the TPU:**
1. Write a 4-bit ripple carry adder
2. Write an 8-bit shift register with load
3. Write a 3-state FSM (IDLE → COMPUTE → DONE)
4. Write a simple dual-port SRAM

---

### 1.3 FPGA Toolchain

Learn your tools before you need them under deadline pressure.

**If using Xilinx (Vivado):**
```
- Create project → Add sources → Set top module
- Run Synthesis → check warnings (undriven signals, latches)
- Run Implementation → check timing report (Fmax)
- Generate Bitstream → Program device
- Use ILA (Integrated Logic Analyzer) for on-chip debugging
```

**If using Intel/Altera (Quartus):**
```
- New project wizard → pin assignment → compile
- TimeQuest Timing Analyzer for Fmax
- SignalTap Logic Analyzer for on-chip debug
```

**Simulation (mandatory):**
```bash
# Icarus Verilog (free, works everywhere)
iverilog -o sim.out top.v testbench.v
vvp sim.out
gtkwave dump.vcd    # view waveforms
```

---

### 1.4 Systolic Array Architecture

This is the core concept. Do not skip understanding this deeply.

**What a systolic array is:**
- A grid of identical Processing Elements (PEs)
- Data flows through the grid in a wave (like a heartbeat — hence "systolic")
- Each PE does: `accumulator += weight × input` every clock cycle

**Weight-stationary dataflow:**
```
1. Load weight matrix W into PEs — each PE holds ONE weight permanently
2. Stream input matrix rows diagonally through the array
3. Partial sums (psums) accumulate vertically down each column
4. After N cycles, column outputs = one row of result matrix C = W × B
```

**The skew pattern (must understand this):**
```
Cycle 0:  b[0][0] enters column 0
Cycle 1:  b[0][1] enters column 0,  b[1][0] enters column 1
Cycle 2:  b[0][2] enters column 0,  b[1][1] enters column 1,  b[2][0] enters column 2
...
```
This diagonal stagger ensures each PE receives the right input at the right time.

---

### 1.5 Fixed-Point and Integer Arithmetic

Your TPU uses INT8 — understanding why and what it costs you is important.

```
INT8 range:     -128 to +127
INT8 multiply:  two 8-bit inputs → 16-bit product (never overflows)
Accumulator:    32-bit to safely accumulate N products without overflow

Quantization:
  float_weight → int8_weight = round(float_weight × scale_factor)
  scale_factor = 127.0 / max(abs(weights))

Quantization error:
  Every weight loses precision. For 4×4 matmul this is negligible.
  For large networks it matters — use quantization-aware training.
```

---

### 1.6 UART Serial Protocol

This is how your PC talks to your FPGA in this project.

```
Protocol basics:
  - Start bit (1 bit low)
  - Data bits (8 bits, LSB first)
  - Stop bit (1 bit high)
  - No clock line — both sides agree on baud rate (115200 bps typical)

What you need to implement in Verilog:
  - UART RX: deserialize incoming bytes, assert done pulse
  - UART TX: serialize bytes to send back to PC

Timing:
  At 115200 baud, one bit = 1/115200 ≈ 8.68 μs
  At 100 MHz clock, one bit = 868 clock cycles
  Your baud rate divider = 100_000_000 / 115200 = 868
```

---

### 1.7 Linear Algebra (just enough)

You need to understand what your hardware is computing and why.

```
Matrix multiply C = A × B:
  C[i][j] = sum over k of A[i][k] × B[k][j]

For a 4×4 case:
  - 4×4 × 4×4 = 64 multiply-accumulate (MAC) operations
  - Result is a 4×4 matrix

Neural network inference (FC layer):
  y = W × x + b
  Where:
    W = weight matrix (4×4, preloaded)
    x = input vector (4×1, your data)
    b = bias vector (4×1, added after matmul)
    y = output vector (4×1, class scores)

ReLU activation:
  f(x) = max(0, x)   — clamp negatives to zero
```

---

### 1.8 Python for Offline Work

You don't need to be a Python expert. You need these specific skills.

```python
import numpy as np

# What you need to know:
np.matmul(A, B)            # golden model for verification
np.clip(x, -128, 127)      # INT8 clamping
np.round(x).astype(np.int8) # quantization
serial.Serial('/dev/ttyUSB0', 115200)  # UART communication (pyserial)
```

---

## Part 2 — Learning Sequence (Recommended Order)

```
Week 1:  Verilog basics → write PE module → simulate with testbench
Week 2:  Build 4×4 array → test with hardcoded matrices → verify against numpy
Week 3:  Add FSM + weight buffer + activation buffer
Week 4:  Add UART TX/RX → test end-to-end from PC
Week 5:  Add ReLU + output buffer + performance counter
Week 6:  Pick application → train model → run inference → measure accuracy
```

---

## Part 3 — Application Guide: 0–9 Digit Classifier

### 3.1 Overview

Run a 2-layer neural network on your TPU to classify handwritten digits (0–9).
Since your array is 4×4, you compress the MNIST image down to 4 features using PCA.

```
Input:   28×28 pixel image → PCA → 4 float features → quantize → 4 INT8 values
Layer 1: 4×4 weight matrix W1 → matmul → ReLU → 4 INT8 activations
Layer 2: 4×4 weight matrix W2 → matmul → 4 INT8 scores
Output:  argmax of 4 scores → predicted class (maps to digit 0–9 via label encoding)
```

> Note: With only 4 output neurons you can classify 4 distinct classes directly.
> To cover all 10 digits, classify pairs (0–4 and 5–9) in two passes, or use a
> 4-class grouping and report top-group accuracy.

---

### 3.2 Step 1 — Train the Model (Python, run once on your PC)

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Keep only digits 0-3 for a clean 4-class demo
mask = y < 4
X, y = X[mask], y[mask]

# Compress 784 features → 4 via PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Train tiny 2-layer network: 4 → 4 → 4
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu',
                      max_iter=500, random_state=42)
model.fit(X_scaled, y)
print(f"Accuracy: {model.score(X_scaled, y):.2%}")

# Extract and quantize weights to INT8
def quantize(W):
    scale = 127.0 / np.max(np.abs(W))
    return np.clip(np.round(W * scale), -128, 127).astype(np.int8), scale

W1, s1 = quantize(model.coefs_[0])   # shape: (4, 4)
W2, s2 = quantize(model.coefs_[1])   # shape: (4, 4)

print("W1 (INT8):\n", W1)
print("W2 (INT8):\n", W2)
np.save('W1_int8.npy', W1)
np.save('W2_int8.npy', W2)
np.save('pca_components.npy', pca.components_)
np.save('scaler_params.npy', np.array([scaler.mean_, scaler.scale_]))
```

---

### 3.3 Step 2 — Hardcode Weights into Verilog

```verilog
// In your weight buffer initialization
// Replace with actual values from W1 printed above
task load_W1;
  begin
    weight_buf[0][0] = 8'sh23;  // W1[0][0]
    weight_buf[0][1] = 8'shF1;  // W1[0][1]
    // ... all 16 values
  end
endtask
```

---

### 3.4 Step 3 — Two-Layer Inference FSM

```
State machine for 2-layer inference:

IDLE
  │ start pulse
  ▼
LOAD_W1       ← load W1 into weight buffer
  │
  ▼
COMPUTE_L1    ← stream input through array (7 cycles for 4×4)
  │
  ▼
RELU_L1       ← clamp negatives in output buffer (1 cycle)
  │
  ▼
LOAD_W2       ← swap W2 into weight buffer
  │
  ▼
COMPUTE_L2    ← feed L1 output as new activations (7 cycles)
  │
  ▼
OUTPUT        ← send 4-byte result over UART
  │
  ▼
IDLE
```

---

### 3.5 Step 4 — Python UART Communication Script

```python
import serial
import numpy as np

# Load saved transforms
pca_comp   = np.load('pca_components.npy')
mean_, std_ = np.load('scaler_params.npy')

def preprocess_image(img_flat):
    """img_flat: 784-element array, pixel values 0-255"""
    x = img_flat / 255.0
    x_pca = pca_comp @ x          # project to 4D
    x_scaled = (x_pca - mean_) / std_
    return np.clip(np.round(x_scaled * 64), -128, 127).astype(np.int8)

def run_inference(port, img_flat):
    ser = serial.Serial(port, 115200, timeout=2)
    features = preprocess_image(img_flat)

    # Send 4 input bytes
    ser.write(bytes(features.tobytes()))

    # Read 4 output bytes (class scores)
    result = ser.read(4)
    scores = np.frombuffer(result, dtype=np.int8)
    predicted_class = np.argmax(scores)
    print(f"Scores: {scores} → Predicted digit: {predicted_class}")
    ser.close()
    return predicted_class
```

---

### 3.6 What to Report

| Metric | How to Measure |
|--------|---------------|
| Inference accuracy | Run 100 test images, count correct predictions |
| Cycles per inference | Performance counter (2 × matmul cycles + overhead) |
| Latency (ms) | cycles / Fmax |
| Weight quantization error | `mean(abs(W1_float - W1_int8 / scale))` |

---

## Part 4 — Application Guide: Signal Classifier

### 4.1 Overview

Classify a 4-sample signal snippet into one of 4 categories:
**Sine**, **Square**, **Noise**, **DC (constant)**.

This ties directly to your ECE signal processing background and needs zero external datasets.

```
Input:   4 consecutive amplitude samples (INT8, range -128 to 127)
Layer 1: 4×4 weight matrix → matmul → ReLU
Layer 2: 4×4 weight matrix → matmul
Output:  argmax → {0: Sine, 1: Square, 2: Noise, 3: DC}
```

---

### 4.2 Step 1 — Generate Synthetic Training Data (Python)

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

np.random.seed(42)
N = 2000   # samples per class

def gen_sine():
    phase = np.random.uniform(0, 2*np.pi)
    freq  = np.random.uniform(0.1, 0.4)
    t = np.arange(4)
    return np.sin(2 * np.pi * freq * t + phase)

def gen_square():
    phase = np.random.uniform(0, 2*np.pi)
    freq  = np.random.uniform(0.1, 0.4)
    t = np.arange(4)
    return np.sign(np.sin(2 * np.pi * freq * t + phase))

def gen_noise():
    return np.random.uniform(-1, 1, 4)

def gen_dc():
    level = np.random.uniform(-1, 1)
    return np.full(4, level) + np.random.normal(0, 0.05, 4)

# Build dataset
generators = [gen_sine, gen_square, gen_noise, gen_dc]
X = np.vstack([np.array([g() for _ in range(N)]) for g in generators])
y = np.repeat([0, 1, 2, 3], N)

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Train
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu',
                      max_iter=1000, random_state=42)
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.2%}")

# Quantize
def quantize(W):
    scale = 127.0 / np.max(np.abs(W))
    return np.clip(np.round(W * scale), -128, 127).astype(np.int8)

W1 = quantize(model.coefs_[0])
W2 = quantize(model.coefs_[1])
print("W1:\n", W1)
print("W2:\n", W2)
```

---

### 4.3 Step 2 — Sending Signal Samples Over UART

```python
import serial
import numpy as np

CLASS_NAMES = ['Sine', 'Square', 'Noise', 'DC']

def classify_signal(port, samples_float):
    """
    samples_float: list of 4 floats in range [-1.0, 1.0]
    """
    # Quantize input to INT8
    samples_int8 = np.clip(np.round(np.array(samples_float) * 127),
                           -128, 127).astype(np.int8)

    ser = serial.Serial(port, 115200, timeout=2)
    ser.write(bytes(samples_int8.tobytes()))

    result = ser.read(4)
    scores = np.frombuffer(result, dtype=np.int8)
    predicted = np.argmax(scores)
    print(f"Input:  {samples_float}")
    print(f"Scores: {scores}")
    print(f"Class:  {CLASS_NAMES[predicted]}")
    ser.close()
    return predicted

# Example: test with a known sine wave
t = np.arange(4)
sine_samples = list(np.sin(2 * np.pi * 0.2 * t))
classify_signal('/dev/ttyUSB0', sine_samples)
```

---

### 4.4 Live Demo Script (type samples manually)

```python
import serial, numpy as np

CLASS_NAMES = ['Sine', 'Square', 'Noise', 'DC']
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)

while True:
    raw = input("Enter 4 samples separated by spaces (or q to quit): ")
    if raw.strip() == 'q':
        break
    vals = list(map(float, raw.split()))
    if len(vals) != 4:
        print("Need exactly 4 values")
        continue
    samples = np.clip(np.round(np.array(vals) * 127), -128, 127).astype(np.int8)
    ser.write(bytes(samples.tobytes()))
    scores = np.frombuffer(ser.read(4), dtype=np.int8)
    print(f"→ {CLASS_NAMES[np.argmax(scores)]}  (scores: {scores})\n")

ser.close()
```

---

### 4.5 What to Report

| Metric | Notes |
|--------|-------|
| Classification accuracy | Test on 200 synthetic samples per class |
| Confusion matrix | Shows which signal types get confused most |
| Cycles per inference | From performance counter |
| Robustness to noise | Add Gaussian noise to clean inputs, retest |

---

## Part 5 — Quick Reference: Common Bugs and Fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| Using `=` in clocked block | Simulation passes, synthesis fails | Use `<=` always in `always @(posedge clk)` |
| Accumulator overflow | Wrong output values | Widen accumulator to 32 bits |
| Missing skew on inputs | Diagonal garbage in output | Stagger input feeding by column index |
| Weight not loaded before compute | FSM goes to COMPUTE too early | Gate the COMPUTE transition on a `weights_ready` flag |
| UART baud mismatch | Garbage received | Verify divider = `clk_freq / baud_rate` |
| Signed/unsigned mismatch | Negative values treated as large positive | Declare all data paths as `signed` consistently |

---

## Part 6 — Recommended Free Resources

| Topic | Resource |
|-------|---------|
| Verilog basics | HDLBits (hdlbits.01xz.net) — exercises with auto-grading |
| Systolic arrays | Original Kung & Leiserson 1978 paper (short, readable) |
| FPGA tools | Xilinx Vivado Quick Start (docs.xilinx.com) |
| UART protocol | UART tutorial — nandland.com |
| INT8 quantization | Google's quantization whitepaper (arxiv 1712.05877) |
| Python + serial | PySerial documentation (pyserial.readthedocs.io) |

---

*Guide prepared for academic TPU mini-project — 4×4 Weight-Stationary Systolic Array on FPGA*

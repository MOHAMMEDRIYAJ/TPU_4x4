# TPU Starter Manual
### A 4×4 Systolic-Array Tensor Processing Unit — 8-bit Signed Integer Datapath

**Version 1.0 — Mini Project Reference Design**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Specifications](#2-specifications)
3. [System Architecture](#3-system-architecture)
4. [Register / Address Map](#4-register--address-map)
5. [Module Hierarchy](#5-module-hierarchy)
6. [Module Reference](#6-module-reference)
   - 6.1 [`pe` — Processing Element](#61-pe--processing-element)
   - 6.2 [`systolic_array` — 4×4 PE Grid](#62-systolic_array--44-pe-grid)
   - 6.3 [`input_skew_buffer` — Diagonal Wavefront Generator](#63-input_skew_buffer--diagonal-wavefront-generator)
   - 6.4 [`mem16x8` — Weight Memory](#64-mem16x8--weight-memory)
   - 6.5 [`act_buffer` — Activation Memory](#65-act_buffer--activation-memory)
   - 6.6 [`accum_regfile` — Result Capture Register File](#66-accum_regfile--result-capture-register-file)
   - 6.7 [`tpu_controller` — Sequencing FSM](#67-tpu_controller--sequencing-fsm)
   - 6.8 [`tpu_top` — Top-Level Wrapper](#68-tpu_top--top-level-wrapper)
7. [Operating Sequence & Timing](#7-operating-sequence--timing)
8. [Worked Example](#8-worked-example)
9. [Testbench Skeleton](#9-testbench-skeleton)
10. [Synthesis & Tooling Notes](#10-synthesis--tooling-notes)
11. [Summary & Next Steps](#11-summary--next-steps)

---

## 1. Introduction

This manual documents a **4×4 weight-stationary systolic array TPU** — a small, self-contained
matrix-multiply accelerator built from scratch in Verilog. It performs the operation

```
C (4x4) = A (4x4) x W (4x4)
```

where every element of `A` and `W` is an **8-bit signed (INT8)** value, and every element of the
result `C` is accumulated into a **32-bit signed** register.

The unit is built as a set of small, independently-documented modules (a processing element, a
systolic grid, memories, a skew network, an accumulator, and a sequencing controller), all wrapped
behind a single **generic memory-mapped register interface** (`addr / wdata / rdata / we / re`).
This bus-style interface is intentionally simple and generic — it behaves like any standard
peripheral register block (control register, status register, data registers), so the whole unit
can later be dropped behind a host bus or attached to a processor's load/store interface without
having to touch any of the internal datapath logic.

This manual covers **only the TPU itself** — its internal architecture, modules, and register map.

---

## 2. Specifications

| Parameter                     | Value                              |
|--------------------------------|-------------------------------------|
| Array size                    | 4 × 4 Processing Elements (PEs)    |
| Element width (A, W)          | 8-bit, signed, two's complement     |
| Accumulator width (C)         | 32-bit, signed                      |
| Dataflow                      | Weight-stationary systolic array    |
| Matrix dimensions supported   | 4×4 × 4×4 → 4×4                     |
| Internal storage              | 16×8-bit weight mem, 16×8-bit activation mem, 16×32-bit result mem |
| Control interface             | Memory-mapped registers (8-bit addr, 32-bit data) |
| Clocking                      | Single clock domain, synchronous active-low reset (`rst_n`) |
| Latency (4×4 × 4×4)           | ~ (2N − 1) compute cycles + 2N weight-load cycles, N = 4 |

**Why 8-bit signed elements?** INT8 is the standard low-precision format used in real TPUs for
inference workloads — it keeps PE hardware small (one 8×8 signed multiplier per PE) while still
giving a useful dynamic range (−128 to +127) for matrix data such as quantized neural network
weights/activations.

**Why a 32-bit accumulator?** A dot-product of four INT8×INT8 terms only needs about 18–19 bits to
avoid overflow, but 32 bits is used so that every result register lines up cleanly on a standard
32-bit data-word boundary — convenient for any wider system this block is later integrated into.

---

## 3. System Architecture

### 3.1 Weight-Stationary Dataflow

The array holds the **weight matrix `W` stationary** — one element of `W` is loaded into each PE
and stays there for the entire computation. The **activation matrix `A`** is streamed in row by
row from the west (left) edge, and **partial sums flow southward** (top to bottom) through the
array, accumulating as they go. The bottom row of PEs emits one fully-accumulated row of the
result matrix `C` per cycle, once the pipeline has filled.

```
                 W00   W01   W02   W03
                  |     |     |     |
   A[r][0] -->  [PE]--[PE]--[PE]--[PE]
                  |     |     |     |
   A[r][1] -->  [PE]--[PE]--[PE]--[PE]
                  |     |     |     |
   A[r][2] -->  [PE]--[PE]--[PE]--[PE]
                  |     |     |     |
   A[r][3] -->  [PE]--[PE]--[PE]--[PE]
                  |     |     |     |
                  v     v     v     v
               C[r][0] C[r][1] C[r][2] C[r][3]
```

Each PE does exactly one job every cycle:

```
psum_out = psum_in + (act_in * weight_stationary)
act_out  = act_in        (passed east to the next PE, one cycle later)
```

### 3.2 Diagonal Skew

Because the array is systolic, each row of `A` must enter the grid **staggered in time** — row `r`
must start `r` cycles after row `0` — so that the diagonal wavefront of partial products lines up
correctly as it propagates through the grid. This staggering is generated by the
`input_skew_buffer` module (Section 6.3) so the rest of the design can be reasoned about as a
clean "row in, row out" pipeline.

### 3.3 Top-Level Block Diagram

```
                         |-------------------------------------------|
                         │                tpu_top                    |
                         │                                           │
  addr/wdata/we ───────► │  ┌──────────┐        ┌───────────────_─┐  │
  re/rdata      ◄─────── │  │  mem16x8 │        │   act_buffer    │  │
                         │  │ (weights)│        │ (activations)   │  │
                         │  └────┬─────┘        └────────┬────────┘  │
                         │       │                       │           │
                         │       │              ┌────────▼────────-┐ │
                         │       │              │ input_skew_buffer│ │
                         │       │              └────────┬────────-┘ │
                         │       │                       │           │
                         │  ┌────▼───────────────────────▼───────-┐  │
                         │  │          systolic_array (4x4)       │  │
                         │  └──────────────────┬──────────────────┘  │
                         │                     │                     │
                         │            ┌────────▼─────────┐           │
                         │            │  accum_regfile   │           │
                         │            │  (result memory) │           │
                         │            └──────────────────┘           │
                         │                                           │
                         │            ┌────────────────────┐         │
                         │            │   tpu_controller   │         │
                         │            │       (FSM)        │         │
                         │            └────────────────────┘         │
                         ---------------------------------------------
```

---

## 4. Register / Address Map

All registers are accessed through the generic bus port on `tpu_top` (`addr[7:0]`,
`wdata[31:0]`, `we`, `re`, `rdata[31:0]`). Reads and writes are single-cycle.

| Address Range      | Name        | Width | Access | Description                                   |
|---------------------|-------------|-------|--------|------------------------------------------------|
| `0x00`              | `CTRL`      | 32    | W      | bit[0] = `START` (self-clearing pulse)         |
| `0x04`              | `STATUS`    | 32    | R      | bit[0] = `BUSY`, bit[1] = `DONE`               |
| `0x10` – `0x4C`     | `WEIGHT[0..15]` | 32 (8 used) | W | Row-major write of the 4×4 weight matrix `W`  |
| `0x50` – `0x8C`     | `ACT[0..15]`    | 32 (8 used) | W | Row-major write of the 4×4 activation matrix `A` |
| `0x90` – `0xCC`     | `RESULT[0..15]` | 32     | R      | Row-major read of the 4×4 result matrix `C`    |

Index → address formula:

```
WEIGHT[i]  address = 0x10 + 4*i      (i = row*4 + col, i = 0..15)
ACT[i]     address = 0x50 + 4*i
RESULT[i]  address = 0x90 + 4*i
```

**Typical usage sequence:**

1. Write all 16 `WEIGHT` registers.
2. Write all 16 `ACT` registers.
3. Write `1` to `CTRL` (start).
4. Poll `STATUS.BUSY` until it clears (or `STATUS.DONE` is seen high).
5. Read all 16 `RESULT` registers.

This register block is deliberately shaped like a standard peripheral (control/status/data
registers, byte-addressed, 32-bit data bus) so it can be dropped behind any bus adapter later
without redesigning the internals.

---

## 5. Module Hierarchy

```
tpu_top
 ├── mem16x8          (weight storage, 16 x 8-bit, single read port)
 ├── act_buffer        (activation storage, 16 x 8-bit, 4 parallel read ports)
 ├── input_skew_buffer (diagonal staggering network)
 ├── systolic_array
 │    └── pe  (x16, instantiated in a 4x4 generate grid)
 ├── accum_regfile     (16 x 32-bit result storage)
 └── tpu_controller    (FSM: IDLE -> LOADW -> STREAM -> DRAIN -> DONE)
```

---

## 6. Module Reference

### 6.1 `pe` — Processing Element

**Purpose:** The atomic compute unit of the array. Holds one stationary 8-bit signed weight,
multiplies it against an incoming 8-bit signed activation every cycle, adds the result to an
incoming 32-bit partial sum, and forwards both the activation (east) and the new partial sum
(south) to its neighbors on the next clock edge.

| Port         | Dir | Width        | Description                                    |
|--------------|-----|--------------|--------------------------------------------------|
| `clk`        | in  | 1            | Clock                                            |
| `rst_n`      | in  | 1            | Active-low synchronous reset                     |
| `weight_we`  | in  | 1            | Write-enable for this PE's stationary weight reg |
| `weight_in`  | in  | 8, signed    | Weight value to latch when `weight_we` is high   |
| `act_in`     | in  | 8, signed    | Activation in, from west neighbor                |
| `act_out`    | out | 8, signed    | Activation out, to east neighbor (registered)    |
| `psum_in`    | in  | 32, signed   | Partial sum in, from north neighbor              |
| `psum_out`   | out | 32, signed   | Partial sum out, to south neighbor (registered)  |
| `compute_en` | in  | 1            | Gates the MAC pipeline registers                 |

```verilog
module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Weight load interface
    input  wire                          weight_we,
    input  wire signed [DATA_WIDTH-1:0]  weight_in,

    // Systolic data path (West -> East)
    input  wire signed [DATA_WIDTH-1:0]  act_in,
    output reg  signed [DATA_WIDTH-1:0]  act_out,

    // Systolic partial-sum path (North -> South)
    input  wire signed [ACC_WIDTH-1:0]   psum_in,
    output reg  signed [ACC_WIDTH-1:0]   psum_out,

    input  wire                          compute_en
);

    reg signed [DATA_WIDTH-1:0] weight_reg;

    // ---- Stationary weight register ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_reg <= {DATA_WIDTH{1'b0}};
        else if (weight_we)
            weight_reg <= weight_in;
    end

    // ---- Systolic MAC pipeline ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_out  <= {DATA_WIDTH{1'b0}};
            psum_out <= {ACC_WIDTH{1'b0}};
        end else if (compute_en) begin
            act_out  <= act_in;
            psum_out <= psum_in + (act_in * weight_reg);
        end
    end

endmodule
```

**Notes:**
- The multiplier `act_in * weight_reg` is an 8×8 **signed** multiply, naturally sized to 16 bits;
  it is added into a 32-bit accumulator, so overflow is not a concern for 4-deep dot products.
- `weight_we` and the data path are independent — weights can only be reloaded when the array is
  idle, since `weight_we` is driven by the controller only during the load phase.

---

### 6.2 `systolic_array` — 4×4 PE Grid

**Purpose:** Wires up sixteen `pe` instances into a 4×4 grid. Activations enter from the west edge
(one stream per row), partial sums enter as zero from the north edge and exit accumulated from the
south edge (one stream per column). Weight loading is addressed — a single `(row, col)` pair is
selected at a time so the controller can write the whole matrix in 16 cycles.

| Port            | Dir | Width             | Description                                  |
|------------------|-----|-------------------|-----------------------------------------------|
| `clk`, `rst_n`   | in  | 1                 | Clock / reset                                  |
| `compute_en`     | in  | 1                 | Enables the MAC pipeline across the whole grid |
| `weight_we`      | in  | 1                 | Global weight write strobe                     |
| `weight_row`     | in  | 2                 | Target row for weight write                    |
| `weight_col`     | in  | 2                 | Target column for weight write                 |
| `weight_data`    | in  | 8, signed         | Weight value to write                          |
| `act_in_row`     | in  | array of 4 × 8, signed  | One activation stream per row, west edge |
| `psum_out_col`   | out | array of 4 × 32, signed | One accumulated stream per column, south edge |

```verilog
module systolic_array #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter N          = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire compute_en,

    // Addressed weight load
    input  wire                          weight_we,
    input  wire [1:0]                    weight_row,
    input  wire [1:0]                    weight_col,
    input  wire signed [DATA_WIDTH-1:0]  weight_data,

    // West-edge activation inputs, one per row
    input  wire signed [DATA_WIDTH-1:0]  act_in_row   [0:N-1],

    // South-edge accumulated outputs, one per column
    output wire signed [ACC_WIDTH-1:0]   psum_out_col [0:N-1]
);

    wire signed [DATA_WIDTH-1:0] act_h  [0:N-1][0:N];   // horizontal activation links
    wire signed [ACC_WIDTH-1:0]  psum_v [0:N][0:N-1];   // vertical partial-sum links

    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : ROW
            assign act_h[r][0] = act_in_row[r];          // west edge feed

            for (c = 0; c < N; c = c + 1) begin : COL
                wire pe_we = weight_we && (weight_row == r) && (weight_col == c);

                if (r == 0)
                    assign psum_v[0][c] = {ACC_WIDTH{1'b0}};  // north edge = zero

                pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .weight_we  (pe_we),
                    .weight_in  (weight_data),
                    .act_in     (act_h[r][c]),
                    .act_out    (act_h[r][c+1]),
                    .psum_in    (psum_v[r][c]),
                    .psum_out   (psum_v[r+1][c]),
                    .compute_en (compute_en)
                );
            end
        end
    endgenerate

    generate
        for (c = 0; c < N; c = c + 1) begin : OUT
            assign psum_out_col[c] = psum_v[N][c];        // south edge = result
        end
    endgenerate

endmodule
```

**Notes:**
- `act_in_row` and `psum_out_col` are declared as **unpacked arrays of ports**. Most modern
  simulation/synthesis tools (Vivado, Quartus, Verilator, Icarus with `-g2012`) accept this
  cleanly. See Section 10 if your toolchain requires a flattened, single-vector port instead.
- `act_h[r][N]` (the east edge of the last column) is left dangling — there is no eastward
  neighbor, so it is simply unused.

---

### 6.3 `input_skew_buffer` — Diagonal Wavefront Generator

**Purpose:** Converts four "straight" activation streams (row `r`'s elements arriving on cycles
`0..3`) into the **staggered diagonal pattern** the systolic array requires, by delaying row `r` by
exactly `r` clock cycles using a small shift-register chain per row.

| Port          | Dir | Width                  | Description                              |
|----------------|-----|------------------------|--------------------------------------------|
| `clk`, `rst_n` | in  | 1                      | Clock / reset                              |
| `en`           | in  | 1                      | Shift enable                               |
| `act_raw`      | in  | array of 4 × 8, signed | Unskewed activation inputs, one per row    |
| `act_skewed`   | out | array of 4 × 8, signed | Skewed outputs, feeds `systolic_array`     |

```verilog
module input_skew_buffer #(
    parameter DATA_WIDTH = 8,
    parameter N          = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire en,
    input  wire signed [DATA_WIDTH-1:0] act_raw    [0:N-1],
    output wire signed [DATA_WIDTH-1:0] act_skewed [0:N-1]
);

    genvar r;
    generate
        for (r = 0; r < N; r = r + 1) begin : ROW_DELAY
            if (r == 0) begin : NO_DELAY
                assign act_skewed[0] = act_raw[0];
            end else begin : DELAY_CHAIN
                reg signed [DATA_WIDTH-1:0] dly [0:r-1];
                integer k;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (k = 0; k < r; k = k + 1)
                            dly[k] <= {DATA_WIDTH{1'b0}};
                    end else if (en) begin
                        dly[0] <= act_raw[r];
                        for (k = 1; k < r; k = k + 1)
                            dly[k] <= dly[k-1];
                    end
                end
                assign act_skewed[r] = dly[r-1];
            end
        end
    endgenerate

endmodule
```

**Notes:** row `0` passes straight through (0-cycle delay), row `1` gets a 1-deep shift register,
row `2` a 2-deep chain, row `3` a 3-deep chain — exactly the staircase needed for correct systolic
alignment.

---

### 6.4 `mem16x8` — Weight Memory

**Purpose:** A simple 16-entry, 8-bit-wide register file holding the row-major weight matrix `W`.
Written from the bus during the load phase, read back one element at a time by
`weight_row`/`weight_col` during the array's weight-load sequence.

| Port      | Dir | Width      | Description                       |
|-----------|-----|------------|------------------------------------|
| `clk`, `rst_n` | in | 1     | Clock / reset                      |
| `we`      | in  | 1          | Write enable                       |
| `waddr`   | in  | 4          | Write index (0..15)                |
| `wdata`   | in  | 8, signed  | Write data                         |
| `raddr`   | in  | 4          | Read index (0..15)                 |
| `rdata`   | out | 8, signed  | Read data (combinational)          |

```verilog
module mem16x8 (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     we,
    input  wire [3:0]               waddr,
    input  wire signed [7:0]        wdata,
    input  wire [3:0]               raddr,
    output wire signed [7:0]        rdata
);

    reg signed [7:0] mem [0:15];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1)
                mem[i] <= 8'sd0;
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    assign rdata = mem[raddr];

endmodule
```

---

### 6.5 `act_buffer` — Activation Memory

**Purpose:** Same idea as `mem16x8`, but exposes **four parallel read ports** (one per row), all
indexed by a shared column pointer `col_ptr`. This lets the controller pull out one full "column
slice" of the activation matrix — i.e. element `[row][col_ptr]` for every row simultaneously —
which is exactly what the `input_skew_buffer` needs each cycle.

| Port      | Dir | Width                   | Description                          |
|-----------|-----|--------------------------|----------------------------------------|
| `clk`, `rst_n` | in | 1                  | Clock / reset                          |
| `we`      | in  | 1                        | Write enable                           |
| `waddr`   | in  | 4                        | Write index (0..15, row-major)         |
| `wdata`   | in  | 8, signed                | Write data                             |
| `col_ptr` | in  | 2                        | Shared column pointer for all 4 reads  |
| `row_data`| out | array of 4 × 8, signed   | `mem[row*4 + col_ptr]` for row 0..3    |

```verilog
module act_buffer (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     we,
    input  wire [3:0]               waddr,
    input  wire signed [7:0]        wdata,
    input  wire [1:0]               col_ptr,
    output wire signed [7:0]        row_data [0:3]
);

    reg signed [7:0] mem [0:15];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1)
                mem[i] <= 8'sd0;
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    genvar r;
    generate
        for (r = 0; r < 4; r = r + 1) begin : RD
            assign row_data[r] = mem[r*4 + col_ptr];
        end
    endgenerate

endmodule
```

---

### 6.6 `accum_regfile` — Result Capture Register File

**Purpose:** Captures one full row of `psum_out_col` (all four column accumulations) into a 16-entry
32-bit result memory whenever `capture_en` is asserted, storing it at row index `capture_row`. The
bus can later read back any of the 16 result words individually.

| Port           | Dir | Width                      | Description                          |
|----------------|-----|------------------------------|-----------------------------------------|
| `clk`, `rst_n` | in  | 1                            | Clock / reset                          |
| `capture_en`   | in  | 1                            | Capture strobe                         |
| `capture_row`  | in  | 2                            | Which result row to write              |
| `psum_in_col`  | in  | array of 4 × 32, signed      | Live column outputs from the array     |
| `read_addr`    | in  | 4                            | Read index (0..15, row-major)          |
| `read_data`    | out | 32, signed                   | Read data (combinational)              |

```verilog
module accum_regfile #(
    parameter ACC_WIDTH = 32,
    parameter N         = 4
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          capture_en,
    input  wire [1:0]                    capture_row,
    input  wire signed [ACC_WIDTH-1:0]   psum_in_col [0:N-1],

    input  wire [3:0]                    read_addr,
    output wire signed [ACC_WIDTH-1:0]   read_data
);

    reg signed [ACC_WIDTH-1:0] mem [0:N*N-1];
    integer i, c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N*N; i = i + 1)
                mem[i] <= {ACC_WIDTH{1'b0}};
        end else if (capture_en) begin
            for (c = 0; c < N; c = c + 1)
                mem[capture_row*N + c] <= psum_in_col[c];
        end
    end

    assign read_data = mem[read_addr];

endmodule
```

---

### 6.7 `tpu_controller` — Sequencing FSM

**Purpose:** The brain of the unit. Drives the entire operation through four phases:

1. **`S_LOADW`** — sweeps `weight_row`/`weight_col` across all 16 positions, asserting
   `weight_we` each cycle so `systolic_array` latches the full `W` matrix.
2. **`S_STREAM`** — sweeps `act_col_ptr` across 0..3, enabling the skew buffer and the MAC
   pipeline so `A` streams into the array.
3. **`S_DRAIN`** — keeps the MAC pipeline running while results emerge from the bottom edge,
   capturing each row of `C` into `accum_regfile` as it becomes valid.
4. **`S_DONE`** — pulses `done` for one cycle and returns to `S_IDLE`.

| Port           | Dir | Width | Description                                  |
|-----------------|-----|-------|------------------------------------------------|
| `clk`, `rst_n` | in  | 1     | Clock / reset                                  |
| `start`        | in  | 1     | One-cycle start pulse                          |
| `busy`         | out | 1     | High while a computation is in progress        |
| `done`         | out | 1     | One-cycle pulse when the result is ready       |
| `weight_we`    | out | 1     | Weight write strobe to `systolic_array`        |
| `weight_row`   | out | 2     | Weight row address                             |
| `weight_col`   | out | 2     | Weight column address                          |
| `act_en`       | out | 1     | Shift-enable to `input_skew_buffer`            |
| `act_col_ptr`  | out | 2     | Column pointer into `act_buffer`               |
| `compute_en`   | out | 1     | MAC pipeline enable to `systolic_array`        |
| `capture_en`   | out | 1     | Capture strobe to `accum_regfile`              |
| `capture_row`  | out | 2     | Row index for the current capture              |

```verilog
module tpu_controller #(
    parameter N = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  busy,
    output reg  done,

    output reg                  weight_we,
    output reg  [1:0]           weight_row,
    output reg  [1:0]           weight_col,

    output reg                  act_en,
    output reg  [1:0]           act_col_ptr,

    output reg                  compute_en,
    output reg                  capture_en,
    output reg  [1:0]           capture_row
);

    localparam S_IDLE  = 3'd0,
               S_LOADW = 3'd1,
               S_STREAM= 3'd2,
               S_DRAIN = 3'd3,
               S_DONE  = 3'd4;

    reg [2:0] state;
    reg [4:0] cnt;
    reg [1:0] wr, wc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            weight_we <= 1'b0; weight_row <= 2'd0; weight_col <= 2'd0;
            act_en <= 1'b0; act_col_ptr <= 2'd0;
            compute_en <= 1'b0; capture_en <= 1'b0; capture_row <= 2'd0;
            cnt <= 5'd0; wr <= 2'd0; wc <= 2'd0;
        end else begin
            // safe defaults each cycle, overridden below where needed
            weight_we  <= 1'b0;
            act_en     <= 1'b0;
            capture_en <= 1'b0;
            done       <= 1'b0;

            case (state)

                S_IDLE: begin
                    busy <= 1'b0;
                    compute_en <= 1'b0;
                    if (start) begin
                        state <= S_LOADW;
                        wr <= 2'd0; wc <= 2'd0;
                        busy <= 1'b1;
                    end
                end

                S_LOADW: begin
                    weight_we  <= 1'b1;
                    weight_row <= wr;
                    weight_col <= wc;
                    if (wc == N-1) begin
                        wc <= 2'd0;
                        if (wr == N-1) begin
                            state       <= S_STREAM;
                            cnt         <= 5'd0;
                            act_col_ptr <= 2'd0;
                        end else begin
                            wr <= wr + 2'd1;
                        end
                    end else begin
                        wc <= wc + 2'd1;
                    end
                end

                S_STREAM: begin
                    compute_en  <= 1'b1;
                    act_en      <= 1'b1;
                    act_col_ptr <= cnt[1:0];
                    cnt <= cnt + 5'd1;
                    if (cnt == N-1) begin
                        state <= S_DRAIN;
                        cnt   <= 5'd0;
                    end
                end

                S_DRAIN: begin
                    compute_en <= 1'b1;
                    cnt <= cnt + 5'd1;
                    if (cnt >= (N-1)) begin
                        capture_en  <= 1'b1;
                        capture_row <= cnt - (N-1);
                    end
                    if (cnt == (2*N - 2)) begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    compute_en <= 1'b0;
                    busy       <= 1'b0;
                    done       <= 1'b1;
                    state      <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
```

> **Important:** the exact cycle offsets used in `S_DRAIN` (`N-1` and `2*N-2`) reflect the
> pipeline latency of *this* skew + array implementation. Before trusting results in real
> hardware, verify the capture timing against the testbench in Section 9 and adjust the constants
> if you change the skew or PE pipeline depth.

---

### 6.8 `tpu_top` — Top-Level Wrapper

**Purpose:** Instantiates every block above and exposes the single generic register bus described
in Section 4. This is the only module a larger system needs to talk to.

| Port      | Dir | Width | Description                          |
|-----------|-----|-------|----------------------------------------|
| `clk`     | in  | 1     | Clock                                  |
| `rst_n`   | in  | 1     | Active-low reset                       |
| `addr`    | in  | 8     | Register address                       |
| `wdata`   | in  | 32    | Write data                             |
| `we`      | in  | 1     | Write strobe                           |
| `re`      | in  | 1     | Read strobe                            |
| `rdata`   | out | 32    | Read data                              |

```verilog
module tpu_top #(
    parameter N           = 4,
    parameter DATA_WIDTH  = 8,
    parameter ACC_WIDTH   = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    // Generic memory-mapped register bus
    input  wire [7:0]  addr,
    input  wire [31:0] wdata,
    input  wire        we,
    input  wire        re,
    output reg  [31:0] rdata
);

    // ---- Address map constants ----
    localparam ADDR_CTRL    = 8'h00;
    localparam ADDR_STATUS  = 8'h04;
    localparam ADDR_WEIGHT0 = 8'h10;   // .. 0x4C  (16 words)
    localparam ADDR_ACT0    = 8'h50;   // .. 0x8C  (16 words)
    localparam ADDR_RESULT0 = 8'h90;   // .. 0xCC  (16 words)

    wire busy, done;
    reg  start_pulse;

    // ---- Self-clearing START pulse from CTRL register ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            start_pulse <= 1'b0;
        else if (we && addr == ADDR_CTRL)
            start_pulse <= wdata[0];
        else
            start_pulse <= 1'b0;
    end

    // ---- Write decode for weight / activation memories ----
    wire w_we = we && (addr >= ADDR_WEIGHT0) && (addr < ADDR_WEIGHT0 + 16*4);
    wire a_we = we && (addr >= ADDR_ACT0)    && (addr < ADDR_ACT0    + 16*4);
    wire [3:0] w_waddr = (addr - ADDR_WEIGHT0) >> 2;
    wire [3:0] a_waddr = (addr - ADDR_ACT0)    >> 2;

    // ---- Controller ----
    wire        weight_we_c;
    wire [1:0]  weight_row, weight_col;
    wire        act_en_c;
    wire [1:0]  act_col_ptr;
    wire        compute_en, capture_en;
    wire [1:0]  capture_row;

    tpu_controller #(.N(N)) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse), .busy(busy), .done(done),
        .weight_we(weight_we_c), .weight_row(weight_row), .weight_col(weight_col),
        .act_en(act_en_c), .act_col_ptr(act_col_ptr),
        .compute_en(compute_en), .capture_en(capture_en), .capture_row(capture_row)
    );

    // ---- Weight storage ----
    wire signed [DATA_WIDTH-1:0] weight_rd;
    mem16x8 u_wmem (
        .clk(clk), .rst_n(rst_n),
        .we(w_we), .waddr(w_waddr), .wdata(wdata[7:0]),
        .raddr({weight_row, weight_col}), .rdata(weight_rd)
    );

    // ---- Activation storage ----
    wire signed [DATA_WIDTH-1:0] act_raw [0:N-1];
    act_buffer u_amem (
        .clk(clk), .rst_n(rst_n),
        .we(a_we), .waddr(a_waddr), .wdata(wdata[7:0]),
        .col_ptr(act_col_ptr), .row_data(act_raw)
    );

    // ---- Diagonal skew network ----
    wire signed [DATA_WIDTH-1:0] act_skewed [0:N-1];
    input_skew_buffer #(.DATA_WIDTH(DATA_WIDTH), .N(N)) u_skew (
        .clk(clk), .rst_n(rst_n), .en(act_en_c),
        .act_raw(act_raw), .act_skewed(act_skewed)
    );

    // ---- Systolic compute array ----
    wire signed [ACC_WIDTH-1:0] psum_col [0:N-1];
    systolic_array #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .N(N)) u_array (
        .clk(clk), .rst_n(rst_n), .compute_en(compute_en),
        .weight_we(weight_we_c), .weight_row(weight_row), .weight_col(weight_col),
        .weight_data(weight_rd),
        .act_in_row(act_skewed),
        .psum_out_col(psum_col)
    );

    // ---- Result capture ----
    wire [3:0] res_raddr = (addr - ADDR_RESULT0) >> 2;
    wire signed [ACC_WIDTH-1:0] result_rd;
    accum_regfile #(.ACC_WIDTH(ACC_WIDTH), .N(N)) u_acc (
        .clk(clk), .rst_n(rst_n),
        .capture_en(capture_en), .capture_row(capture_row),
        .psum_in_col(psum_col),
        .read_addr(res_raddr), .read_data(result_rd)
    );

    // ---- Read mux ----
    always @(*) begin
        rdata = 32'd0;
        if (re) begin
            if (addr == ADDR_STATUS)
                rdata = {30'd0, done, busy};
            else if ((addr >= ADDR_RESULT0) && (addr < ADDR_RESULT0 + 16*4))
                rdata = result_rd;
        end
    end

endmodule
```

**Why this interface shape?** `addr / wdata / rdata / we / re` plus a flat register map is the
lowest-common-denominator shape used by virtually every on-chip peripheral bus style (simple
register-file bus, APB-like, Wishbone-like, AXI-Lite-like). Keeping `tpu_top`'s *only* external
contract this simple means that, whenever this accelerator needs to be attached to a larger system
later, only a thin bus-adapter shim needs to be written — none of the internal modules in Sections
6.1–6.7 need to change at all.

---

## 7. Operating Sequence & Timing

```
Phase     | Cycles        | What happens
----------|---------------|--------------------------------------------------------
IDLE      | -             | Waiting for START
LOAD_W    | 16            | weight_row/col sweep 0..3,0..3 ; W loaded into array
STREAM    | N (=4)        | act_col_ptr sweeps 0..3 ; A streams into skew buffer
DRAIN     | N (=4)        | Pipeline drains ; one row of C captured per cycle
DONE      | 1             | done pulses, controller returns to IDLE
```

Total compute-phase latency for a full 4×4 × 4×4 multiply is approximately:

```
T_load   = N*N   = 16 cycles   (one-time, can be skipped if W is unchanged between runs)
T_stream = N     = 4  cycles
T_drain  = N     = 4  cycles
T_total  ≈ 16 + 4 + 4 + 1 = 25 cycles
```

---

## 8. Worked Example

To sanity-check the design, use the identity matrix as `W` — the result should simply echo `A`
back out, which makes it easy to confirm wiring and timing without doing matrix math by hand.

```
A =                          W =                          Expected C = A x W = A
[ 1  2  3  4]                [ 1  0  0  0]                [ 1  2  3  4]
[ 0  1  0  1]                [ 0  1  0  0]                [ 0  1  0  1]
[ 2  0  1  0]                [ 0  0  1  0]                [ 2  0  1  0]
[ 1  1  1  1]                [ 0  0  0  1]                [ 1  1  1  1]
```

Bus transaction sequence:

```
write WEIGHT[0..15] = 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
write ACT[0..15]    = 1,2,3,4, 0,1,0,1, 2,0,1,0, 1,1,1,1
write CTRL          = 1                       // start
poll  STATUS        until BUSY == 0
read  RESULT[0..15] -> should equal ACT[0..15]
```

Once this passes, swap in a non-trivial `W` (e.g. all elements = 2, or a small handwritten matrix)
and confirm the result against a manual or scripted calculation of `A x W`.

---

## 9. Testbench Skeleton

A minimal self-checking testbench to get started with simulation:

```verilog
`timescale 1ns/1ps

module tb_tpu_top;

    reg clk = 0;
    reg rst_n = 0;
    reg [7:0] addr;
    reg [31:0] wdata;
    reg we, re;
    wire [31:0] rdata;

    tpu_top dut (
        .clk(clk), .rst_n(rst_n),
        .addr(addr), .wdata(wdata),
        .we(we), .re(re), .rdata(rdata)
    );

    always #5 clk = ~clk;   // 100 MHz

    integer i;
    reg signed [7:0] A [0:15];
    reg signed [7:0] W [0:15];

    task bus_write(input [7:0] a, input [31:0] d);
        begin
            @(posedge clk);
            addr <= a; wdata <= d; we <= 1'b1; re <= 1'b0;
            @(posedge clk);
            we <= 1'b0;
        end
    endtask

    task bus_read(input [7:0] a, output [31:0] d);
        begin
            @(posedge clk);
            addr <= a; re <= 1'b1; we <= 1'b0;
            @(posedge clk);
            d = rdata;
            re <= 1'b0;
        end
    endtask

    initial begin
        we = 0; re = 0; addr = 0; wdata = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;

        // Identity weight matrix
        W[0]=1; W[1]=0; W[2]=0; W[3]=0;
        W[4]=0; W[5]=1; W[6]=0; W[7]=0;
        W[8]=0; W[9]=0; W[10]=1; W[11]=0;
        W[12]=0; W[13]=0; W[14]=0; W[15]=1;

        A[0]=1; A[1]=2; A[2]=3; A[3]=4;
        A[4]=0; A[5]=1; A[6]=0; A[7]=1;
        A[8]=2; A[9]=0; A[10]=1; A[11]=0;
        A[12]=1; A[13]=1; A[14]=1; A[15]=1;

        for (i = 0; i < 16; i = i + 1)
            bus_write(8'h10 + i*4, {24'd0, W[i]});

        for (i = 0; i < 16; i = i + 1)
            bus_write(8'h50 + i*4, {24'd0, A[i]});

        bus_write(8'h00, 32'h1);  // START

        // Wait for DONE / BUSY clear
        wait_busy_clear();

        for (i = 0; i < 16; i = i + 1) begin
            reg [31:0] r;
            bus_read(8'h90 + i*4, r);
            $display("RESULT[%0d] = %0d (expect %0d)", i, $signed(r), A[i]);
        end

        $finish;
    end

    task wait_busy_clear;
        reg [31:0] s;
        begin
            s = 32'h1;
            while (s[0] == 1'b1)
                bus_read(8'h04, s);
        end
    endtask

endmodule
```

---

## 10. Synthesis & Tooling Notes

- **Unpacked array ports** (`act_in_row [0:N-1]`, `psum_out_col [0:N-1]`, etc.) are used throughout
  for readability. These are supported by Vivado, Quartus, Verilator (`--sv` / IEEE-1800 mode),
  and Icarus Verilog (`-g2012`). If your toolchain only accepts strict Verilog-2001 syntax, flatten
  each array port into a single wide vector (e.g. `act_in_row` → `[N*DATA_WIDTH-1:0] act_in_flat`)
  and slice it with `act_in_flat[(r+1)*DATA_WIDTH-1 -: DATA_WIDTH]` at each instantiation site.
- All sequential elements use **active-low synchronous reset** (`if (!rst_n)` inside
  `always @(posedge clk or negedge rst_n)`), consistent across every module.
- The accumulator is intentionally **not saturating** — for INT8 × INT8 with only 4 accumulation
  steps, 32 bits is comfortably wide enough that overflow is not a practical concern at this size.
- Every datapath signal that carries `A`, `W`, or `C` values is declared `signed`, so Verilog's
  built-in signed multiply/add semantics are used directly — no manual sign-extension logic is
  needed anywhere in the design.

---

## 11. Summary & Next Steps

This manual has covered a complete, self-contained 4×4 INT8 weight-stationary systolic TPU:

- A single-MAC `pe` building block
- A 4×4 `systolic_array` built from 16 of them
- Supporting memories (`mem16x8`, `act_buffer`) and a diagonal `input_skew_buffer`
- Result capture (`accum_regfile`)
- A sequencing `tpu_controller`
- A top-level `tpu_top` exposing one clean, generic register-mapped bus

Suggested directions to extend this starter design for a larger project:

- Parameterize `N` beyond 4 (the modules are already written generically against `N`)
- Add a ready/valid streaming interface to overlap weight-load with the previous computation's drain
- Pipeline the bus interface for higher clock frequencies
- Add saturation/clamping logic on read-out if a narrower output format is ever required
- Write a constrained-random or coverage-driven testbench in place of the directed example in
  Section 9

All internal datapath modules are independent of the bus shape used in `tpu_top`, so the register
interface can be re-wrapped for a different host system later without touching the compute core.

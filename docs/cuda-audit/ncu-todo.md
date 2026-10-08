# CUDA audit: open items that need ncu

State of `feature/nvidia` at ccf26ade2. These need GPU performance counters, which were not available on the hosts used so far (containers with a user namespace and `RmProfilingAdminOnly: 1` on the host give `ERR_NVGPUCTRPERM`, even as root with full capabilities).

## Host requirements

- `cat /proc/self/uid_map` shows `0 0 4294967295` (no user namespace remap).
- `grep RmProfilingAdminOnly /proc/driver/nvidia/params` shows 0, or ncu runs as root on the host itself. Host side fix: `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, then reload the driver.
- A compute capability 12.x card (RTX PRO 2000/4000/5000/6000 Blackwell, GeForce 50). The per-SM results carry over between them, absolute numbers do not (SM count, L2, DRAM bus, power limit). B200/B300 (cc 10.x) do not use the native FP4 path at all (`common.cuh:296` and `:376`), they run NVFP4 through the W4A8 int8 MMA kernels.
- Nsight Compute 2025.1 or newer. On Ubuntu with the CUDA repo: `apt-get install cuda-nsight-compute-13-0` (or the 12-8/12-9 package).

## Setup

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build-cuda -j --target test-backend-ops llama-perplexity llama-bench
g++ -O2 -std=c++17 -o wbench docs/cuda-audit/wbench.cpp -Iggml/include -Lbuild-cuda/bin -lggml -lggml-base -Wl,-rpath,$PWD/build-cuda/bin
./wbench nvfp4 1,2,4,8          # decode, 16 distinct 4096x14336 weights, read from DRAM
./wbench nvfp4 512,2048 16 10   # prefill
GGML_CUDA_MMQ_PREC=q8 ./wbench nvfp4 512,2048 16 10   # force W4A8
```

`WB_M=<rows>` changes the weight rows (default 4096, K is 14336). For ncu, set `GGML_CUDA_DISABLE_GRAPHS=1` and skip the warmup launches, for example:

```sh
GGML_CUDA_DISABLE_GRAPHS=1 ncu --set full -k regex:'mul_mat_q|quantize_mmq' --launch-skip 40 --launch-count 4 -o nvfp4-w4a4 ./wbench nvfp4 2048 16 1
```

## Reference numbers (RTX PRO 6000 Blackwell Server Edition, 600 W, CUDA 12.8, driver 595.91)

Device read ceiling: 1529 GB/s (torch reduction over 2 GiB, spec 1792).

Decode, n = 1, fit of time = fixed + bytes / bandwidth over WB_M = 4096 and 16384:

| type   | stream GB/s | fixed us/op | 4096x14336 us/op |
|--------|-------------|-------------|------------------|
| nvfp4  | 1539        | 3.36        | 24.80            |
| mxfp4  | 1544        | 4.32        | 24.53            |
| q4_0   | 1534        | 3.23        | 24.74            |
| q4_K   | 1526        | 3.45        | 25.08            |
| iq4_xs | 1542        | 3.53        | 23.76            |
| q8_0   | 1540        | 3.63        | 44.16            |
| bf16   | 1540        | 1.97        | 78.22            |

Prefill, 4096x14336, weights in DRAM, TFLOPS at n = 512 / 2048:

| path                        | TFLOPS    |
|-----------------------------|-----------|
| nvfp4 W4A4 (default on 12.x) | 449 / 454 |
| mxfp4 W4A4                  | 398 / 414 |
| nvfp4 W4A8                  | 198 / 192 |
| mxfp4 W4A8                  | 276 / 272 |
| q4_0, q4_K, iq4_xs, q8_0    | 259 - 294 |
| bf16 (cuBLAS)               | 286 / 322 |
| cuBLASLt nvfp4 (torch `_scaled_mm`, M=4096 N=2048 K=14336, no activation quantization) | 1248 |
| cuBLASLt fp8 e4m3 / bf16    | 622 / 364 |

## Open items

### 1. NVFP4 W4A4 prefill is at about 36% of cuBLASLt

Kernels: `mul_mat_q<GGML_TYPE_NVFP4, ..., GGML_PREC_Q4>` (`mmq.cuh:963`), tile loader `ggml_cuda_mmq_load_tiles_nvfp4_nvfp4` (`mmq-load-tiles.cuh:1729`), activation quantizer `quantize_mmq_nvfp4` (`quantize.cu:128`), `mul_mat_q_stream_k_fixup`.

Questions:
- Split of the op time between `quantize_mmq_nvfp4` (scale search per 16 values) and `mul_mat_q`.
- What `mul_mat_q` waits on: long scoreboard on the global loads (at source level the loader does 8 + 1 32-bit loads per 36 byte block, blocks are only 4 byte aligned), barrier (load, sync, MMA with no overlap), MIO throttle or shared memory bank conflicts, register spills.
- Tensor pipe utilization against the 1248 TFLOPS that cuBLASLt reaches on the same shape.

ncu sections (all in `--set full`): WarpStateStats (stall reasons), ComputeWorkloadAnalysis (tensor pipe utilization), MemoryWorkloadAnalysis (shared memory bank conflicts, local memory traffic from spills), Occupancy and LaunchStats (registers, occupancy limits), SourceCounters (stalls per SASS line). Spills are also visible at build time with `-Xptxas -v`.

Compare with upstream PR #28572 (closed, cp.async and TMA tile loads for NVFP4 MMQ, reported +14% pp on an RTX 5090): `git fetch https://github.com/ggml-org/llama.cpp pull/28572/head:pr-28572`. If the stalls are on the loads and #28572 removes them, a weight repack is not needed for prefill either. If the remaining gap is in the MMA or scale path, layout does not matter.

### 2. NVFP4 W4A8 is about 30% slower than Q4_0 W4A8

`GGML_CUDA_MMQ_PREC=q8`, loader `ggml_cuda_mmq_load_tiles_nvfp4` (`mmq-load-tiles.cuh:1673`, `MMQ_DP4A_TXS_Q8_0_16`). Only used on 12.x when forced, but it is the default NVFP4 path on cc < 12.0 (Ada, Hopper, B200/B300). Suspect: one UE4M3 scale per 16 values means twice the scale work of Q4_0 and a conversion per sub-block. Compare instruction mix and issue slots against `mul_mat_q<GGML_TYPE_Q4_0>` at the same shape.

### 3. Fixed cost per mat-vec in decode

3.2 - 4.3 us per op for every quantized type, 2.0 us for bf16, about 14% of a 4096x14336 nvfp4 mat-vec. The weight streaming itself is at the read ceiling, so a weight layout change cannot help decode. Suspect: the separate `quantize_q8_1` launch plus the ramp and tail of `mul_mat_vec_q`. A kernel trace (`nsys profile --trace=cuda`) answers most of this without counters; ncu (LaunchStats waves per SM, SourceCounters) for the tail.

### 4. K-quant MMVQ drops at n = 2..4

Not NVFP4, found in the same sweep (GB/s at n = 1 / 2 / 4 / 8): q4_K 1317 / 1136 / 815 / 1041, q5_K 1318 / 997 / 818 / 1115. NVFP4, Q4_0, IQ4_XS and Q6_K stay at 1220 - 1340 up to n = 4. On this GPU Q4_K uses MMVQ up to n = 5 and Q5_K up to n = 6, so n = 8 is MMQ. Profile `mul_mat_vec_q<GGML_TYPE_Q4_K>` at ncols 1, 2, 4 (occupancy, registers, stall reasons).

### 5. MMVQ at n = 8

Every 4 bit type falls from about 1330 to 940 - 1020 GB/s at n = 8 (q8_0 1144, bf16 1362). Lower priority. Same metrics as item 4.

## Settled without ncu

- Q8_0 CPU/CUDA KLD floor (0.0016) is the sensitivity of q8 activations to small upstream differences, not a kernel bug: flash attention on vs off inside one backend gives the same 0.0016 (BF16: 7.5e-5), with f16 activations (`GGML_CUDA_FORCE_CUBLAS`) it drops to 1.9e-4. Per op CPU and CUDA match to NMSE 1.5e-14 (MMVQ) and 3.5e-8 (MMQ keeps the activation scale in float).
- The same flash attention test gives 0.0015 - 0.0021 for every q8 activation type, so after ccf26ade2 the Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL results (0.0023 - 0.0028) are close to the floor. K-quants, Q6_K and IQ4_XS (0.009 - 0.013) also contain the q8_K vs q8_1 activation format difference.
- On Blackwell, NVFP4 and MXFP4 prefill (n > 8) quantize the activations to FP4 by default, so their CPU/CUDA divergence depends on the ubatch size more than for other types.

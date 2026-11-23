# Inkling 2 Mini Vulkan Backend Optimizations (Experimental)

> **Experimental AI-assisted fork** built on top of Unsloth's `llama.cpp` (`b12588-mix-c0c3de8`).
> Custom Vulkan GLSL shaders and graph fusions in this repository were designed, iterated, and verified using AI coding agents on a Windows 11 laptop (`AMD Radeon 780M Graphics`, 64 GB unified memory), inspired by recent autonomous GPU kernel optimization reports on AMD Strix Halo.

As of commit time, upstream `ggml-org/llama.cpp` and `unslothai/llama.cpp` optimize Inkling 2 Mini on CUDA (`#401`, `#402`, `#407`, `#411`–`#414`), HIP/WMMA (`#406`), and Metal (`#404`), but **do not yet ship native Vulkan `FLASH_ATTN_EXT_BANDED` (including quantized KV) or fused Inkling 2 Vulkan kernels** — causing stock Vulkan builds to split graphs across GPU/CPU (`194 splits`) and fall back heavily when `-ctk q8_0 -ctv q8_0` is enabled.

---

## Key Changes (Single Squashed Feature Commit Over Base)

1. **Native Vulkan `FLASH_ATTN_EXT_BANDED` with Quantized KV (`Q8_0` / `Q4_0` / `F16`) & Per-Query Sinks**
   - Supports `F32`, `F16`, and `BF16` `rel_logits` via bit-packed flags (`REL_TYPE_SHIFT`) on `binding = 7` (`flash_attn_base.glsl`) without extra descriptor bindings or `int16` extensions.
   - Implements native `USE_BANDED` tile culling across scalar (`flash_attn.comp`), CoopMat1 (`flash_attn_cm1.comp`), CoopMat1 Flip (`flash_attn_cm1_flip.comp`, including per-query sink row offset fix), and CoopMat2 (`flash_attn_cm2.comp`) shaders.
   - Removes the temporary `if (banded)` CoopMat2/Flip bypass and enables quantized KV cache decode on Vulkan without F16 fallback bloat.
2. **Fused Inkling 2 Mini Vulkan Shaders & Graph Ops**
   - `depth_mix.comp` (single-pass Depth-Attention & stack mixing; sized for `SubgroupSize >= 8` and guarded by `require_full_subgroups`),
   - `loop_norm_down.comp` + `loop_up_gate.comp` (loop controller norm/down & uniform-subgroup reduction/up-gate),
   - `control_gate.comp` (6-op bidirectional `tanh` `control()` gate fusion),
   - `ssm_conv_dyn.comp` + `ssm_conv.comp` (dynamic ShortConv + residual scaling),
   - `inkling_rel_bias.comp`, `l2_norm.comp` (fused row L2 clip), `swiglu_oai.comp` (SiTU GLU), `topk_moe.comp`, and NVFP4 (`mul_mat_vec_nvfp4.comp`) / `mul_mmq_cm1` improvements.
3. **Tests Added**
   - 16 quantized-KV (`GGML_TYPE_Q8_0`, `GGML_TYPE_Q4_0`) `FLASH_ATTN_EXT_BANDED` cases in `tests/test-backend-ops.cpp`.

---

## Measured Benchmarks (AMD Radeon 780M Graphics, Windows 11)

Model: `Inkling-2-Mini-UD-Q4_K_M.gguf` (`15.46 GiB`, `26.07 B`), `-ngl 99`.
Comparing stock `unslothai/llama.cpp` Vulkan build (`b12588-mix-c0c3de8`) vs. this repository:

### 1. `llama-bench` (`-fa 1 -p 512 -n 128`)

| Test | KV Cache Type | Stock Unsloth Vulkan | This Fork (Vulkan) | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `pp512` (Prefill) | `f16` (`-r 3`) | `95.83 ± 1.90 tok/s` | **`166.63 ± 2.68 tok/s`** | **1.74x** |
| `tg128` (Decode) | `f16` (`-r 3`) | `4.97 ± 0.87 tok/s` | **`5.58 ± 0.47 tok/s`** | **1.12x** |
| `pp512` (Prefill) | `q8_0` (`-r 3` / `-r 1`) | `43.79 ± 3.04` (`-r 3`) / `50.59` (`-r 1`) | **`135.24 ± 1.80 tok/s`** (`-r 3`) | **2.67x – 3.09x** |
| `tg128` (Decode) | `q8_0` | Timed out >360s (`-r 3`) / `0.83 tok/s` (`-r 1`) | **`4.85 ± 0.57 tok/s`** (`-r 3`) | **5.84x** |

### 2. 80K Context with `q8_0` KV Cache (`llama-completion -c 81920 -fa on -ctk q8_0 -ctv q8_0 -fit off -n 32`)

| Metric | Stock Unsloth Vulkan | This Fork (Vulkan) | Delta |
| :--- | ---: | ---: | ---: |
| `Vulkan0 compute buffer size` | `6,236.59 MiB` | **`810.64 MiB`** | **-5,425.95 MiB (-87.0%)** |
| `Vulkan_Host compute buffer size` | `303.02 MiB` | **`86.02 MiB`** | **-71.6%** |
| Graph splits (`pp512 / tg1`) | `194 / 194` | **`1 / 1`** | **Zero CPU fallback** |
| Graph reserve time (`sched_reserve`) | `7,968 ms` | **`1,144 ms`** | **7.0x faster** |
| Process Peak Working Set | `22,091.5 MB` | **`16,432.6 MB`** | **-5.53 GB** |
| Free Vulkan VRAM after load | `22,504 MiB` | **`28,147 MiB`** | **+5,643 MiB (+5.51 GiB)** |
| Cold 16-tok Prompt / 31-tok Eval | `2.35` / `0.62 tok/s` | **`9.34` / `4.78 tok/s`** | **3.97x / 7.71x** |

*(Steady-state `llama-server` generation with `-c 81920 -fa on -kvu -ctk q8_0 -ctv q8_0` reaches `~8.1–10.9 tok/s` on this laptop depending on sustained TDP throttling.)*

---

## Verification Status

Ran on `Vulkan0` (`AMD Radeon 780M Graphics`) using `test-backend-ops` after the final shader/host edits:
- `FLASH_ATTN_EXT_BANDED`: **69 / 69 tests passed** (`f32`, `f16`, `bf16` `rel_logits` + 16 `q8_0`/`q4_0` KV cases)
- `L2_NORM`: **31 / 31 tests passed**
- `SSM_CONV`: **75 / 75 tests passed**
- All 13 custom parity verifier suites (`SITU`, `CLIP`, `TOPK_MOE`, `CONTROL_GATE`, `SSM_CONV_DYN`, `DEPTH_MIX`, `DEPTH_MIX_STACK`, `LOOP_CTRL`, `FA_BANDED` F16/Q8_0/Q4_0, etc.) passed with `0` failures (`cos >= 0.999999`).

---

## Build & Run (Windows / Linux with Vulkan SDK)

```bash
cmake -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench llama-completion test-backend-ops

# Verify Vulkan ops
./build/bin/test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT_BANDED
./build/bin/test-backend-ops test -b Vulkan0 -o L2_NORM
./build/bin/test-backend-ops test -b Vulkan0 -o SSM_CONV

# Run Inkling 2 Mini with quantized KV cache on Vulkan
./build/bin/llama-server \
  -m Inkling-2-Mini-UD-Q4_K_M.gguf \
  -ngl 99 -fa on -kvu -ctk q8_0 -ctv q8_0 -c 81920 --host 127.0.0.1 --port 8080
```

# hy4-exl3-vllm-serving — drop-in overlay for Tencent Hy4-flash EXL3 on DGX Spark (GB10)

This is the **vLLM runtime overlay** that serves [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash-FP8)
on a single NVIDIA DGX Spark GB10 sm_121, through the EXL3 plugin line (`vllm_exl3`), from the
[`vllm/vllm-openai:hy4-flash`](https://hub.docker.com/r/vllm/vllm-openai) container. It pairs with
the **3.0-bit EXL3 pack** [`aitamedia/Hy4-flash-EXL3-3.0bpw`](https://huggingface.co/aitamedia/Hy4-flash-EXL3-3.0bpw)
and our **FP8-native dense** pack build.

Companion material: recipes + runbooks in [`styles01/sparkrun-recipes`](https://github.com/styles01/sparkrun-recipes)
(`recipes/hy4-flash-exl3-vllm-tp1.yaml`), and an experimental **TensorFold CUDA port** for the same
model in [`styles01/tensorfold-hy4-cuda`](https://github.com/styles01/tensorfold-hy4-cuda).

## Measured (GB10, single Spark, TP1, 2026-11-21)

- **16.1 tok/s** coherent single-stream decode (no spec), 240–420 ms TTFT
- **34.3 tok/s aggregate** at c3 concurrency (256K max context)
- speculative decoding (DFly2) currently **net-negative**: acceptance length 1.00 — see "known bugs"

## What this overlay is

`overlay/` is a drop-in tree for `/usr/local/lib/python3.12/dist-packages` **inside the
`vllm/vllm-openai:hy4-flash` image**. Copy it over (or bind-mount ro) and set
`PYTHONPATH=/path/to/overlay`. It contains:

- `vllm_exl3/` — the EXL3 plugin with our patches: FP8-native non-expert routing
  (`VLLM_FLASH_FP8_NONEXPERT=1`, `VLLM_FLASH_FP8_ALL=1`) with the matcher exclusions the GB10
  kernel constraints force (see below), `VLLM_FLASH_FP8_KV`-adjacent fixes, engram disk backend
  (`engram.py`), `compressed_indexer.py` with the once-per-step schedule cache, `exl3.py` loader
  fixes, DFly2 drafter loader fixes (in progress — see bugs).
- `vllm/models/hy_v4{,_flash}/` — the model implementation with GB10 fixes (attention, engram,
  DFly, sm120 attention helpers).
- `vllm/config/speculative.py`, `vllm/v1/worker/gpu/spec_decode/dfly/*` — DFly specular config
  machinery.
- `vllm/transformers_utils/configs/hy_v4.py` — HF config shim.

Base image: `vllm/vllm-openai:hy4-flash`. Any file not present in `overlay/` is unmodified from
that image.

## Hard-won facts (GB10 sm_121)

1. **FP8-blockwise kernel = cutlass only.** torch's DeepSeek-style blockwise `_scaled_mm` is
   SM90-only. Boot with
   `--kernel-config '{"linear_backend":"torch","linear_backend_per_quant":{"fp8_block_w8a8":"cutlass"}}'`
   or weight-load/profile crashes.
2. **Weights that must stay BF16** in an FP8-native pack:
   - `vit.*` (vision tower — loader has no FP8 path)
   - indexer projections (FP32-correctness path)
   - `kv_a_proj_with_mqa` (N=576, not 128-aligned — cutlass rejects)
   - `q_a_proj` + `fused_qkv_a_proj` (the merged layer takes one quant_config; dtype must be uniform)
3. **Spawned EngineCore processes don't inherit PYTHONPATH** (mp spawn env-scrub). The launcher writes
   a `.pth` file (`import sys, os; sys.path.insert(1, ...)`) into image `dist-packages` — that's what
   makes the overlay load everywhere. See launcher.
4. `VLLM_USE_DEEP_GEMM=0` — DeepGEMM JIT crashes during CUDA-graph capture on sm_121.
5. Engram table (63 GB here) streams from disk (`disk_offload: true, disk_threads: 256`) — never
   resident; c3 fits in ~103 GiB with the 3.0-bit pack.

## The FP8-native pack build

Per-shard conversion (scripts in `pack_build/`):

1. Copy the verified EXL3 pack.
2. Convert BF16-stored dense linears (layers >= 1: attention `o_proj/q_b_proj/linear_gate`,
   `kv_b_proj`, mlp gate/up/down, `hidden_correction`, `eh_proj`, merged projs) to torch fp8
   e4m3 with **in-shard `weight_scale_inv`** `(ceil(N/128), ceil(K/128))` block scales
   (`pack_build/convert_pack.py`).
3. Restore the must-stay-BF16 list from the clean copy + drop their scale entries
   (`pack_build/fix_vit.py`, `fix_kva.py`, `fix_qa.py`).

Result: `exl3-3.0bpw-fp8lin`. Without this, every BF16-stored dense linear runs BF16 GEMV
(~90 launches/step on GB10 — ~5 ms/step of pure overhead).

## Known bugs / in-progress

- **DFly2 spec drafter** (`vllm/models/hy_v4/nvidia/dfly.py`): the drafter's compressed-tensors
  fp8 params compute with **init-default scale values** (unit) rather than checkpoint blockwise
  scales → acceptance length 1.00. The MTP draft path (`mtp.py`) already maps `.scale` →
  `.weight_scale_inv`; the same mapping needs to be done for the DFly modules. Status: fix in
  progress on our side; PRs/pings welcome — this is the last blocker for spec.
- If you hit `Unknown Flash checkpoint weight: layers.0.*` style errors: layer 0 uses dense BF16
  attn/mlp in this architecture — the matcher deliberately excludes layer-0 linears from FP8.

## Launcher

`hy4-tp1.sh` (paths environment-specific; keep the argv continuation chain intact — a single
orphaned `\` drops later flags silently). Key boot flags summarized above; full runbook in
sparkrun-recipes.

## Credits

- Tencent Hy team — model, DFly2, reference serving recipe
- vcruz305 / ExLlamaV3 line — the EXL3 vLLM runtime this overlay extends
- atakann — DGX Spark recipes and attention pattern this build inherits
- upsteam vLLM EXL3 plugin authors

License: follow the container image's license terms; all local modifications Apache-2.0 where
applicable.

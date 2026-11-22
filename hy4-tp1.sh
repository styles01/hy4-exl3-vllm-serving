#!/usr/bin/env bash
# Hy4-flash EXL3 3.0bpw TP1 on spark1. Proven atakann attention pattern + exl3 experts.
set -euo pipefail
NAME=hy4-serve
MODEL_DIR=$MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin
RUNTIME=/opt/hy4-serve/site-packages
PORT=8000
docker rm -f $NAME >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/hy4-serve"
cat > $HY4_TMP/hy4-serve.sh <<'EOF'
#!/usr/bin/env bash
set -e
export PYTHONPATH=/opt/hy4-serve/site-packages
export CUTE_DSL_ARCH=sm_121a
export EXL3_FUSED_MOE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_EGDUMP=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TRITON_MLA_SPARSE=1
export VLLM_ENGRAM_ZERO=1
export VLLM_FLASH_FP8_NONEXPERT=1
export VLLM_FLASH_FP8_ALL=1
export VLLM_MOE_GATE_FIRST=1
export VLLM_PROFILE_STEP=32
export VLLM_EXL3_ROUTE_STATS=1
export VLLM_EXL3_ROUTE_STATS_DIR=$MODEL_ROOT/routes
export VLLM_EXL3_ROUTE_STATS_EXPERTS=512
export VLLM_EXL3_ROUTE_STATS_ROWS=48
export VLLM_EXL3_ROUTE_STATS_DUMP_SEC=20
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
echo "import sys, os; sys.path.insert(1, os.environ.get('PYTHONPATH', '/opt/hy4-serve/site-packages'))" > /usr/local/lib/python3.12/dist-packages/zz_oracle.pth
export EXL3_L2PF=attn,moe,next
export EXL3_L2PF_HC=1
export EXL3_ROUTER=1
export EXL3_MOE_COOP_WIDE=1
export VLLM_EXL3_MOE_PARALLEL_GROUPS=1
export EXL3_MROW_GROUPED=1
export EXL3_INT8_GEMV=0
export EXL3_GR_INT8=1
export VLLM_EXL3_COOP=1
export VLLM_EXL3_COALESCE_SHARDS=1
exec python3 -c "import vllm_exl3; vllm_exl3.register(); from vllm.entrypoints.launchers.api_server.entry import main; main()" \
  --model $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin --served-model-name hy4-flash-exl3 --host 0.0.0.0 --port 8000 \
  --quantization exl3 --trust-remote-code --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 262144 --max-num-seqs 3 \
  --max-num-batched-tokens 8192 --kv-cache-dtype auto \
  --kv-cache-memory-bytes 6G --gpu-memory-utilization 0.85 \
  --engram-config '{"cpu_offload":false,"disk_offload":true,"disk_threads":256}' \
  --speculative-config null \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --kernel-config '{"linear_backend":"torch","linear_backend_per_quant":{"fp8_block_w8a8":"cutlass"}}' --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":0}'
EOF
chmod +x $HY4_TMP/hy4-serve.sh
docker run -d --name $NAME --network host --ipc host --shm-size 64g --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 --restart no --entrypoint /bin/bash \
  -v $MODEL_DIR:$MODEL_DIR:ro -v $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin:$MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin:ro -v $RUNTIME:$RUNTIME:ro \
  -v $HY4_TMP/hy4-serve.sh:/serve.sh:ro -v "$HOME/.cache/hy4-serve":/root/.cache -v $MODEL_ROOT/routes:$MODEL_ROOT/routes \
  -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:hy4-flash /serve.sh
echo "launched $NAME on :$PORT"

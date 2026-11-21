import torch
import glob, os, json
from safetensors.torch import save_file, load_file

torch.set_grad_enabled(False)
src = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-clean'
dst = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin'

want_sub = ("routed_expert_down_proj", "routed_expert_up_proj", "shared_experts",
            "kv_b_proj", "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa",
            "self_attn.o_proj", "self_attn.q_b_proj", "self_attn.linear_gate",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "hidden_correction", "eh_proj")

def in_targets(k):
    if "layers." not in k and "dfly_layers." not in k:
        return False
    if "scale" in k:
        return False
    return any(w in k for w in want_sub)

def quant_block(w):
    wf = w.float()
    N, K = wf.shape
    nb, kb = (N + 127) // 128, (K + 127) // 128
    pad = wf.new_zeros(nb * 128, kb * 128)
    pad[:N, :K] = wf
    blocks = pad.view(nb, 128, kb, 128).permute(0, 2, 1, 3).reshape(nb, kb, -1)
    amax = blocks.abs().amax(-1).clamp_min(1e-12)
    scale = (amax / 448.0)
    qb = (blocks.view(nb, kb, 128, 128) / scale.view(nb, kb, 1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    qb = qb.view(nb, kb, 128, 128).permute(0, 2, 1, 3).reshape(nb * 128, kb * 128)[:N, :K]
    return qb.contiguous(), scale.to(torch.bfloat16).contiguous()

os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(src + '/model-*.safetensors'))
if len(files) < 9:
    raise SystemExit("RSYNC-NOT-COMPLETE, %d shard files" % len(files))
print("shards:", len(files), flush=True)
new_scale_keys = {}
for f in files:
    base = os.path.basename(f)
    tmp = os.path.join(dst, base + '.tmp')
    tensors = load_file(f)
    for k, v in list(tensors.items()):
        if in_targets(k) and v.ndim == 2 and v.dtype in (torch.bfloat16, torch.float16):
            q, sc2 = quant_block(v)
            tensors[k] = q
            sk = k[:-len('.weight')] + '.weight_scale_inv'
            tensors[sk] = sc2
            new_scale_keys[sk] = base
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, os.path.join(dst, base))
    print("done", base, "scales:", len(new_scale_keys), flush=True)
    del tensors
for extra in glob.glob(src + '/*'):
    b = os.path.basename(extra)
    p = os.path.join(dst, b)
    if not os.path.exists(p):
        try:
            os.link(extra, p)
        except OSError:
            os.symlink(extra, p)
ipath = dst + '/model.safetensors.index.json'
idx = json.load(open(ipath))
wm = idx['weight_map']
wm.update(new_scale_keys)
json.dump(idx, open(ipath, 'w'), indent=1)
print("BUILD-DONE scales:", len(new_scale_keys), flush=True)

import torch, glob, json
from safetensors.torch import load_file
torch.set_grad_enabled(False)
src = '$MODEL_ROOT/Hy4-flash/exl3-3.0bpw-clean'
bad = []
for f in sorted(glob.glob(src + '/model-*.safetensors')):
    t = load_file(f)
    for k, v in t.items():
        if k.endswith('.weight') and v.dtype == torch.bfloat16 and v.ndim == 2:
            n, kk = v.shape
            if kk % 128 or n % 128:
                bad.append((k, str(v.shape)))
    del t
print("non-128-aligned:", len(bad))
for b in bad[:24]:
    print(b)


# skip non-2d guard handled in loop

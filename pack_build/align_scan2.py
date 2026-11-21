import torch, glob
from safetensors.torch import load_file
torch.set_grad_enabled(False)
src = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin'
bad = []
for f in sorted(glob.glob(src + '/model-*.safetensors')):
    t = load_file(f)
    for k, v in t.items():
        if k.endswith('.weight') and v.dtype == torch.float8_e4m3fn and v.ndim == 2:
            n, kk = v.shape
            if kk % 128 or n % 128:
                bad.append((k, str(v.shape)))
    del t
print("misaligned fp8:", len(bad))
for b in bad[:20]:
    print(b[0].split('model.')[-1], b[1])

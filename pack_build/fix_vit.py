import torch, glob, os
from safetensors.torch import save_file, load_file

torch.set_grad_enabled(False)
clean = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-clean'
dst = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin'
restored = 0
for clean_f in sorted(glob.glob(clean + '/model-*.safetensors')):
    src_t = load_file(clean_f)
    out = dst + '/' + os.path.basename(clean_f)
    if not os.path.exists(out):
        continue
    tensors = load_file(out)
    changed = False
    for k, v in src_t.items():
        if 'vit.' in k and k in tensors and tensors[k].dtype == torch.float8_e4m3fn:
            tensors[k] = v
            removed_key = k[:-len('.weight')] + '.weight_scale_inv'
            if removed_key in tensors:
                del tensors[removed_key]
            changed = True
            restored += 1
    if changed:
        tmp = out + '.tmp'
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, out)
        print("restored vit in", os.path.basename(clean_f), flush=True)
    del src_t, tensors
print("done", restored, flush=True)

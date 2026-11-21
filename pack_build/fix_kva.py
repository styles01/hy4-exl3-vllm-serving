import torch, glob, os
from safetensors.torch import save_file, load_file
torch.set_grad_enabled(False)
clean = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-clean'
dst = '/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin'
restored = 0
for clean_f in sorted(glob.glob(clean + '/model-*.safetensors')):
    src_t = load_file(clean_f)
    out = dst + '/' + os.path.basename(clean_f)
    tensors = load_file(out)
    changed = False
    for k, v in src_t.items():
        if 'kv_a_proj_with_mqa' in k and v.ndim == 2 and tensors.get(k) is not None and tensors[k].dtype == torch.float8_e4m3fn:
            tensors[k] = v
            sk = k[:-len('.weight')] + '.weight_scale_inv'
            if sk in tensors:
                del tensors[sk]
            changed = True
            restored += 1
    if changed:
        tmp = out + '.tmp'
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, out)
        del src_t, tensors
        print("restored kv_a in", os.path.basename(clean_f), flush=True)
    else:
        del src_t, tensors
print("done", restored, flush=True)

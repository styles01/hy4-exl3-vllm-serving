import subprocess
inner = r'''
import subprocess as sp
r = sp.run(["bash", "-c", "mkdir -p /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-clean && nohup rsync -a --info=progress2 -e 'ssh -o BatchMode=yes' /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw/ jaita@10.200.0.2:/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-clean/ > /tmp/rsync_pack.log 2>&1 & echo RSYNC-START"], capture_output=True, text=True, timeout=60)
print((r.stdout or '')[-200:], r.stderr[-120:])
'''
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "jaita@192.168.2.185", "python3 -"], input=inner, capture_output=True, text=True, timeout=110)
print(r.stdout[-260:], r.stderr[-100:])

import subprocess, re
env = open("/Users/clawdio/.hermes/.env").read()
pw = re.match(r"^SUDO_PASSWORD=(\S+)$", [l for l in env.splitlines() if l.startswith("SUDO_PASSWORD=")][0]).group(1)
cmd = ("printf '%s\\n' '" + pw + "' | sudo -S -p '' sh -c '"
       "mkdir -p /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin && "
       "mount -t nfs4 -o ro,vers=4.2,noatime 10.200.0.2:/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin 2>/dev/null; "
       "mountpoint -q /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin || mount -t nfs4 -o ro,vers=4.2,noatime 10.200.0.2:/home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin; "
       "ls /home/jaita/MODELS/Hy4-flash/exl3-3.0bpw-fp8lin | wc -l'")
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "jaita@192.168.2.185", cmd], capture_output=True, text=True, timeout=180)
print(r.stdout[-300:], r.stderr[-150:])

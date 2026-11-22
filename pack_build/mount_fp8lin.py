import subprocess, re
env = open("/Users/clawdio/.hermes/.env").read()
pw = re.match(r"^SUDO_PASSWORD=(\S+)$", [l for l in env.splitlines() if l.startswith("SUDO_PASSWORD=")][0]).group(1)
cmd = ("printf '%s\\n' '" + pw + "' | sudo -S -p '' sh -c '"
       "mkdir -p $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin && "
       "mount -t nfs4 -o ro,vers=4.2,noatime 10.200.0.2:$MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin 2>/dev/null; "
       "mountpoint -q $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin || mount -t nfs4 -o ro,vers=4.2,noatime 10.200.0.2:$MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin; "
       "ls $MODEL_ROOT/Hy4-flash/exl3-3.0bpw-fp8lin | wc -l'")
r = subprocess.run(["ssh", "-o", "BatchMode=yes", "<spark-host>", cmd], capture_output=True, text=True, timeout=180)
print(r.stdout[-300:], r.stderr[-150:])

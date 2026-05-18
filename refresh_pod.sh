#!/bin/bash
# Run this INSIDE the pod's web terminal (RunPod console -> Connect -> Web Terminal).
#
# The pod's boot script writes v1 cloud_swap.py from an embedded base64 on EVERY container restart,
# then `exec python cloud_swap.py`. So a simple file replace gets reverted as soon as the python
# process dies (PID 1 death = container restart).
#
# Trick: install a sitecustomize.py hook in python's site-packages. sitecustomize is auto-imported
# at python startup. The hook checks if we're launching cloud_swap.py and re-execs into the v2
# file we fetched from github. Boot script writes v1, python loads, hook intercepts, v2 runs.
# This survives stop/start cycles.

set -e
cd /workspace

REPO=https://raw.githubusercontent.com/bluedoorclips/easy-face-swap-cloud/main

echo "=== 1. Pull v2 source from github ==="
wget -q -O /workspace/cloud_swap_v2.py $REPO/cloud_swap.py
wget -q -O /workspace/prompts_library.py $REPO/prompts_library.py
mkdir -p /workspace/app
cp /workspace/prompts_library.py /workspace/app/prompts_library.py
wget -q -O /workspace/app/train_lora.py $REPO/train_lora.py 2>/dev/null || true
ls -l /workspace/cloud_swap_v2.py /workspace/prompts_library.py
echo

echo "=== 2. Install sitecustomize.py hook ==="
PYDIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Installing hook into $PYDIR/sitecustomize.py"
cat > $PYDIR/sitecustomize.py <<'EOF'
# Intercept cloud_swap.py launches and re-exec into v2 from /workspace/cloud_swap_v2.py.
# Self-updates v2 from github on each boot so stop/start picks up latest.
import sys, os

def _looks_like_swap_entry():
    if not sys.argv:
        return False
    a0 = sys.argv[0]
    return a0.endswith("cloud_swap.py") and "cloud_swap_v2" not in a0

if _looks_like_swap_entry() and not os.environ.get("VISO_V2_LOADED"):
    os.environ["VISO_V2_LOADED"] = "1"
    print("[sitecustomize] intercepting cloud_swap.py -> v2", flush=True)
    try:
        import urllib.request
        REPO = "https://raw.githubusercontent.com/bluedoorclips/easy-face-swap-cloud/main"
        for url, dest in [
            (f"{REPO}/cloud_swap.py",       "/workspace/cloud_swap_v2.py"),
            (f"{REPO}/prompts_library.py",  "/workspace/prompts_library.py"),
            (f"{REPO}/prompts_library.py",  "/workspace/app/prompts_library.py"),
            (f"{REPO}/train_lora.py",       "/workspace/app/train_lora.py"),
        ]:
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                data = urllib.request.urlopen(url, timeout=30).read()
                open(dest, "wb").write(data)
                print(f"[sitecustomize] fetched {dest} ({len(data)} B)", flush=True)
            except Exception as e:
                print(f"[sitecustomize] fetch fail {url}: {e}", flush=True)
    except Exception as e:
        print(f"[sitecustomize] pre-fetch error: {e}", flush=True)
    print("[sitecustomize] os.execvp -> /workspace/cloud_swap_v2.py", flush=True)
    os.execvp(sys.executable, [sys.executable, "/workspace/cloud_swap_v2.py"])
EOF
ls -l $PYDIR/sitecustomize.py
echo

echo "=== 3. Trigger reload by killing PID 1 (container will restart via Docker) ==="
echo "PID 1 process tree:"
ps -ef | grep -E "PID|python|bash" | head -10
echo
echo "After kill, container restarts, boot script re-runs, sitecustomize.py intercepts,"
echo "v2 loads. Expect ~30s downtime then v2 live at https://jywxu3zjhtesms-7860.proxy.runpod.net"
echo
echo "Sending SIGTERM to PID 1 in 3 seconds (Ctrl+C to abort)..."
sleep 3
kill 1
echo "kill sent — connection may drop. Wait 30s and refresh the UI."

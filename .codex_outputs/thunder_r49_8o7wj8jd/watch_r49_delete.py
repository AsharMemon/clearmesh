import json, pathlib, subprocess, time, datetime, sys
OUT = pathlib.Path('/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r49_8o7wj8jd')
LOG = OUT / 'watch_py.log'
TNR = '/Users/Ashar/.tnr/bin/tnr'
START = time.time()
MAX_SECONDS = 4 * 60 * 60

def log(msg):
    with LOG.open('a') as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")
        f.flush()

def run(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)

log(f'start max_seconds={MAX_SECONDS}')
while True:
    elapsed = time.time() - START
    if elapsed > MAX_SECONDS:
        log(f'hard timeout elapsed={elapsed:.0f}; copy/delete')
        break
    status = run(f'{TNR} status --json', timeout=60).stdout
    if '"status": "RUNNING"' not in status:
        log('no running instance; exiting')
        sys.exit(0)
    done = run("ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'test -f /workspace/R49_DONE'", timeout=20)
    if done.returncode == 0:
        log('R49_DONE; copy/delete')
        break
    proc = run("ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'pgrep -af \"run_r49_depth|run_canary.py.*depth_camera_r49\"'", timeout=20)
    if proc.returncode == 0:
        log(f'still running elapsed={elapsed:.0f}s')
        time.sleep(60)
        continue
    log(f'process not found rc={proc.returncode}; copy/delete')
    break

run(f'rsync -az -e ssh tnr-1:/workspace/dualprim_depth_r49.log {OUT}/', timeout=120)
run(f'rsync -az -e ssh tnr-1:/workspace/dualprim_k008_depth_camera_r49/ {OUT}/dualprim_k008_depth_camera_r49/', timeout=300)
log('copied outputs; deleting instance')
run(f'{TNR} delete 0 --yes', timeout=120)
log('delete issued')

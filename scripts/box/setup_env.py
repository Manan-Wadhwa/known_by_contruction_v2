"""Box-1 lease setup, 2026-09-07. Writes /marimo/v2/setup.log; ends with SETUP DONE.

Differences from the snapshot's /marimo/v2/setup_env.py, deliberately:
  * installs with sys.executable -m pip (the snapshot used bare `pip`, which
    lands in /usr/local and only reaches the uv venv because the venv happens
    to set include-system-site-packages=true);
  * importlib.invalidate_caches() after the install, so the same process can
    import what it just installed;
  * timestamps every line, verifies the snapshot after downloading it, and
    fails LOUDLY (SETUP FAILED + traceback) instead of silently omitting the
    marker.
"""
import datetime, importlib, json, os, subprocess, sys, time, traceback

LOG = "/marimo/v2/setup.log"
REPO_ID = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.py", "merges.txt", "vocab.json"]

os.makedirs("/marimo/v2", exist_ok=True)
_log = open(LOG, "a", buffering=1)


def log(msg):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
    _log.write(f"[{ts}] {msg}\n")
    _log.flush()
    os.fsync(_log.fileno())


try:
    t_start = time.time()
    log("=== SETUP START (scripts/box/setup_env.py) ===")
    log(f"python {sys.executable} {sys.version.split()[0]}  pid {os.getpid()}  HOME={os.environ.get('HOME')}")

    # PERSISTENCE (2026-09-08). A lease restore keeps /marimo and nothing else:
    # the image's venv and ~/.cache come back empty. So the model cache lives at
    # /marimo/hf (HF_HOME) and the two packages the image lacks live at
    # /marimo/pylib, installed --no-deps because their dependencies (torch,
    # transformers, accelerate, safetensors, huggingface_hub) are in the venv
    # and a --target install without --no-deps would pull torch into pylib and
    # shadow the venv's build through PYTHONPATH. /marimo/persist/env.sh carries
    # the three exports; every launcher sources it.
    for d in ("/marimo/hf", "/marimo/pylib", "/marimo/persist"):
        os.makedirs(d, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/marimo/hf")
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    if "/marimo/pylib" not in sys.path:
        sys.path.insert(0, "/marimo/pylib")
    with open("/marimo/persist/env.sh", "w") as fh:
        fh.write("export HF_HOME=/marimo/hf\nexport HF_HUB_CACHE=/marimo/hf/hub\n"
                 "export PYTHONPATH=/marimo/pylib${PYTHONPATH:+:$PYTHONPATH}\n")
    log("persistent paths: HF_HOME=/marimo/hf  pylib=/marimo/pylib  env=/marimo/persist/env.sh")
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
           "--no-deps", "--target", "/marimo/pylib", "peft==0.20.0", "hf_transfer==0.1.9",
           # accelerate: present in the 08-27 image's venv, absent from the cleaner image the
           # 2026-09-09 leases came up on; peft imports it. Its own deps are in both images.
           "accelerate==1.14.0"]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip():
            log("  pip| " + line)
    log(f"pip rc={r.returncode}")
    if r.returncode != 0:
        raise SystemExit("pip install failed")

    importlib.invalidate_caches()
    import peft, accelerate, hf_transfer, huggingface_hub, torch, transformers
    log(f"peft {peft.__version__} @ {peft.__file__}")
    log(f"accelerate {accelerate.__version__}  hf_transfer {hf_transfer.__version__}")
    log(f"huggingface_hub {huggingface_hub.__version__}  torch {torch.__version__}  transformers {transformers.__version__}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub import constants as hf_constants
    log(f"HF_HUB_ENABLE_HF_TRANSFER=1  cache={hf_constants.HF_HUB_CACHE}")

    t0 = time.time()
    log(f"snapshot_download {REPO_ID} @ {REVISION}")
    path = snapshot_download(REPO_ID, revision=REVISION, allow_patterns=PATTERNS)
    dt = time.time() - t0
    log(f"MODEL at {path}  in {dt:.1f}s")

    # --- verify what actually landed -------------------------------------
    names = sorted(os.listdir(path))
    st = [n for n in names if n.endswith(".safetensors")]
    total = sum(os.path.getsize(os.path.join(path, n)) for n in names
                if os.path.isfile(os.path.join(path, n)))
    log(f"snapshot files={len(names)} safetensors={len(st)} bytes={total}")
    log("files: " + ", ".join(names))

    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            wm = json.load(fh)["weight_map"]
        shards = sorted(set(wm.values()))
        missing = [s for s in shards if not os.path.exists(os.path.join(path, s))]
        log(f"index lists {len(shards)} shards, missing={missing}")
        if missing:
            raise SystemExit("shards missing from snapshot")
    else:
        log("WARNING: no model.safetensors.index.json")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION, local_files_only=True)
    log(f"tokenizer ok vocab={len(tok)}  ids up/down/left/right="
        f"{[tok.encode(w, add_special_tokens=False) for w in ('up','down','left','right')]}")

    log(f"total {time.time() - t_start:.1f}s")
    log("SETUP DONE")
except BaseException:
    log("SETUP FAILED")
    for line in traceback.format_exc().splitlines():
        log("  !| " + line)
    raise

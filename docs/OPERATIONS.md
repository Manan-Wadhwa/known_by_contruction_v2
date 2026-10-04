# Operations — the runbook

Written 2026-09-09 from the scripts in `scripts/box/`, `scripts/sync_to_sandbox.sh`
and `scripts/pull_from_sandbox.sh`, the fresh-lease procedure in
`experiments/v2/MZ01_maze_world/README.md`, and the failure ledger in
`docs/FAILURES.md`. It is the sequence for taking a rented GPU box from nothing
to a scored result, and for getting back what is recoverable when something
dies. Every command is one that exists in this repository — all twenty box
helpers are under `scripts/box/` and §9 documents each with its usage line;
nothing here depends on a session scratchpad any more. Read `docs/FAILURES.md`
§B before the first launch of a session — every trap in §8 below cost real time
at least once.

Standing environment for everything below:

```
export MZ01_BOX_DIR=/path/to/dir      # holds box1.url and box1.token, mode 600
export BOX=box1                        # which pointer pair to use
B=scripts/box
M=/marimo/repo/experiments/v2/MZ01_maze_world/results
```

**The local machine.** Everything except the GPU stages runs here: the test
suite, the three scorers, the pullers and the watchers. The interpreter is the
repository's own `.venv`, and what is in it is frozen in
**`requirements-local.txt`** at the repository root: **Python 3.14.7, torch
2.13.0+cpu, numpy 2.5.2, pytest 9.1.1** and their dependencies, pinned to the
version each was measured at on 2026-09-09. It is CPU-only on purpose — nothing
local needs a GPU, and the scorers are stdlib-only, so a bare Python 3 runs
them.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m pytest -q                    # 271 passed, 1 skipped, ~10 min
```

The **boxes** have a different environment and it is not this file: their own
venv (torch 2.11.0+cu130, transformers 5.14.1) plus the three packages
`setup_env.py` installs into `/marimo/pylib` (`peft 0.20.0`, `hf_transfer
0.1.9`, `accelerate 1.14.0`). The header of `requirements-local.txt` records
both.

---

## 1. Renting and registering a box

**Where a box comes from.** A box is a **molab sandbox** — marimo's hosted
notebook service — rented by the project owner from their own account. Every
lease so far has come up on an RTX PRO 6000 Blackwell with 96–102 GB. The
sandbox's URL and token come from the owner and are pasted into the pointer
files below; **there is no API in this repository for renting, renewing or
listing one**, and no script here can create a box. When the leases are gone,
the work stops until the owner opens more.

**What a lease is worth knowing.** It is finite, the cap is 12 hours, and it
dies **without warning** well inside that. The shortest deaths in this record:
2026-08-29's box 1, launched ~06:15Z and gone by ~08:37–09:02Z — about 2.5 hours
(`RESULTS_LOG.md` §4.1) — and 2026-09-08's box 1, launched 12:25Z and dead at
~15:27Z with the navigator-v2 training at step 1700 (`docs/FAILURES.md` B4).
One lease was extended in place the same day (15:54 → 17:47Z), and four ran the
day out and expired overnight with finished work on them (B5). A renewed lease
restores an *old* snapshot of the volume, asynchronously, so nothing left on a
dead box comes back. Plan every job to produce a pulled artefact inside the
shortest lease you have seen, not the longest (§6 below).

A box is reached only through its URL and token. **Both live outside the
repository**, in one directory named by `$MZ01_BOX_DIR`, one pair of files per
box:

```
$MZ01_BOX_DIR/box1.url      the sandbox URL
$MZ01_BOX_DIR/box1.token    the token, mode 600
$MZ01_BOX_DIR/box2.url  …   one pair per lease held at once
```

`chmod 600` both. Nothing in `scripts/box/` writes them, none of them is ever
committed, and every helper reads them through `${BOX:-box1}`. Note one
interface hazard that is the repo script's and not a choice: `pull_once.sh`
passes the token to `pull_from_sandbox.sh` as `--token`, so it is visible in
`ps` on the **local** machine for the life of each call.

First contact, which is also the test that the pointer pair is right:

```
bash $B/ex.sh -c "print('alive')"
```

---

## 2. Bring-up

### 2.1 Set the box up, every time

```
bash $B/ex.sh - < $B/setup_env.py
```

Detached; it writes `/marimo/v2/setup.log` and ends with `SETUP DONE` (or
`SETUP FAILED` and a traceback). Poll for it — **do not** trust the log that is
already there:

```
bash $B/ex.sh -c "print(open('/marimo/v2/setup.log').read()[-1500:])"
```

**Every lease since 2026-08-27 has come up on the same recycled snapshot**:
stale code under `/marimo/repo` with no `.git`, HH01-era logs and adapters, no
`peft`, no model cache, and a `setup.log` whose last line says "done" with a
timestamp from 08-27. Nothing it claims is true until you have done it again.

What `setup_env.py` persists, and why it matters: a lease restore keeps
`/marimo` and nothing else, so the model cache goes to `/marimo/hf`
(`HF_HOME`), the packages the image lacks go to `/marimo/pylib` (on
`PYTHONPATH`), and the three exports land in `/marimo/persist/env.sh`, which
**every launcher must source**. The install is
`peft==0.20.0 hf_transfer==0.1.9 accelerate==1.14.0`, `--no-deps --target
/marimo/pylib` — `--no-deps` because a `--target` install without it would pull
torch into `pylib` and shadow the venv's build through `PYTHONPATH`.

**The two images seen so far.** The 08-27 image's venv has `accelerate`; the
cleaner image the 2026-09-09 leases came up on does **not**, and `peft` imports
it — which failed four setups until `accelerate` was added to the persistent
install (`b648562`, `docs/FAILURES.md` B12). Neither image has `peft`,
`hf_transfer` or the weights. Assume nothing is installed.

Setup also verifies what landed: file count, safetensors count, every shard
named in `model.safetensors.index.json` present, and a tokenizer round-trip of
the four move words. A missing shard raises rather than being discovered by a
run.

**On a fresh lease, do the whole of §2 with one call per box:**

```
for b in box1 box2 box3 box4; do bash $B/bringup2.sh $b > $MZ01_BOX_DIR/bringup_$b.log 2>&1 & done; wait
```

`bringup2.sh <boxN>` wipes `/marimo/pylib`, runs `setup_env.py` and waits for
`SETUP DONE`, pushes the three binaries (§2.2), syncs the code at `HEAD` (§2.3),
and md5-checks **six** files box against worktree (§2.4: `maze_orgb.py`,
`mz01_explore_trial.py`, `mz01_orgb_driver.py`, `run_orgb.py`, `navigator.py`,
`mz01_navigator_driver.py`). It ends `[boxN <time>Z] ready`. Exit 2 is a failed
setup, exit 3 an md5 mismatch. It launches nothing.

**The `setup FAILED` false positive, and the follow-up.** Trap 5 below (B11)
fired again on 2026-09-09: all four fresh leases reported
`setup FAILED` between **17:11:15Z and 17:11:28Z** while every box's own
`/marimo/v2/setup.log` had already logged `SETUP DONE` — box 1's at
**17:11:14Z**, one second before the check's line. The setup had succeeded on
all four; only `bringup2.sh`'s single post-hoc read of the log had not seen it.
**Never act on that line without reading the box's own log:**

```
bash $B/ex.sh -c "print(open('/marimo/v2/setup.log').read()[-400:])"   # look for SETUP DONE
```

If it says `SETUP DONE`, run the follow-up, which is the same script with the
setup step skipped:

```
bash $B/bringup2_nosetup.sh <boxN>
```

That is what recovered all four boxes at 17:12:25Z, each ending
`MD5 MATCH (6 files) at 64e4abb` and `ready` by 17:13:03Z — three minutes,
against a re-download of the weights. `bringup2_nosetup.sh` skips only the
setup; the pushes, the sync and the md5 check are identical, so nothing is
taken on trust that `bringup2.sh` would have verified.

### 2.2 The three pushes

```
bash $B/push_file.sh experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.pt   $M/navigator/nav_nav48.pt
bash $B/push_file.sh experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.json $M/navigator/nav_nav48.json
bash $B/push_file.sh experiments/v2/MZ01_maze_world/results/orgb/warm_start_s0.pt    $M/orgb/warm_start.pt
```

Chunked, offset-addressed, sha256-verified on both sides, ~1 MB/s, about 20
minutes for the two adapters. Offset-addressed means a re-run **resumes**: the
box appends a chunk only when the remote size equals that chunk's offset, so an
interrupted push is finished rather than restarted and a double-send cannot
duplicate bytes. Each ends with `PUSH_VERIFIED <path> <bytes> <sha256> …`; any
other last line is a failure.

The JSON sidecar is not optional: it carries the role glyphs the organism stage
must render with, and `manifest.adapter_sha256` in it is checked against the
adapter file. A sidecar from a different navigator run would hand the organism a
cue glyph its resident adapter never saw.

For a run on a different navigator, push that navigator's `.pt`/`.json` pair and
its own warm start (`results/orgb_nav48u/warm_start.pt`, not
`results/orgb/warm_start.pt`).

**Three things about `push_file.sh` that bite.**

1. **The argument order is `<local> <remote>`** — the opposite of the pullers,
   which take `<box> <remote> <local>`. `push_file.sh` takes no box argument at
   all: it reads `${BOX:-box1}` from the environment, so `export BOX=box3`
   before the call is what aims it, and a forgotten `BOX` sends the file to box
   1 in silence.
2. **The resume is offset-addressed, so a *changed* file must be removed
   remotely before a re-push.** The box appends a chunk only when the remote
   file's current size equals that chunk's offset. That is what makes an
   interrupted push resumable, and it is also what makes a re-push into a
   *different* file of the same name wrong: if the remote copy is shorter, the
   push appends the tail of the new file onto the head of the old one and fails
   the whole-file sha at the end (`PUSH_FAIL sha/size mismatch`); if it is the
   same length, nothing is sent and the sha fails on the first try; if it is
   longer, the script refuses outright with `PUSH_FAIL remote is larger`. In
   every case the fix is the same — clear the remote copy first:

   ```
   PUSH_RESTART=1 bash $B/push_file.sh <local> <remote>    # truncates the remote file, then pushes
   ```

   or delete the remote path by hand through `ex.sh`. Retrying the plain push
   cannot recover; it re-aims at the same wrong offset.
3. **Only `PUSH_VERIFIED` is success.** The last line is
   `PUSH_VERIFIED <remote> <bytes> <sha256> <seconds> <MB/s>`; anything else —
   including a mid-run `remote size drifted to N; re-aiming` that never reaches
   the end — is a failure, and the exit codes distinguish them (2 no local file,
   3 unreadable remote size, 4 remote larger, 5 remote overshot, 6 six
   consecutive chunk failures, 1 sha/size mismatch).

### 2.3 Sync the code, at a CLEAN commit

```
git status --porcelain -- src experiments tests scripts | grep -v '/results/'   # must be empty
MARIMO_URL=$(cat $MZ01_BOX_DIR/$BOX.url) MARIMO_TOKEN_FILE=$MZ01_BOX_DIR/$BOX.token \
  scripts/sync_to_sandbox.sh src/calibration/maze_orgb.py scripts/mz01_explore_trial.py \
                             scripts/score_mz01_explore.py experiments/v2/MZ01_maze_world/run_orgb.py
```

Name every file you expect to land; the script prints `SYNC_CHECK <file>
PRESENT|MISSING` for each and exits non-zero on a MISSING. Note the
`SYNC_OK 0.1.0 <sha> <n> py files` line and **keep that sha** — it is what goes
on the launch line as `CALIBRATION_GIT_SHA`.

If the tree is dirty the sha is suffixed `-dirty` and the script warns, listing
the files. That is not cosmetic: `runner.git_sha()` falls back to
`CALIBRATION_GIT_SHA`, so a run of uncommitted code would otherwise record a
clean sha whose contents are not the code that ran. Fourteen arm files in this
tree carry `04e5e69-dirty` for exactly that reason (`RESULTS_LOG.md` §5).
**Sync from a clean tree**; if
you cannot, the `-dirty` marker is what makes the record honest, not an excuse
to skip it.

### 2.4 The md5 check, against the commit the box was synced at

Verify that the code on the box is the code you think it is, **for every file
the run executes**, comparing against the *commit* rather than the working tree:

```
REF_SHA=<the SYNC_OK sha, without -dirty>
FILES="src/calibration/maze_orgb.py src/calibration/maze2.py src/calibration/navigator.py \
       scripts/mz01_explore_trial.py scripts/score_mz01_explore.py \
       experiments/v2/MZ01_maze_world/run_orgb.py"

# remote, prefixed so the hashes can be picked out of a stream that also carries warnings
python3 - $FILES <<'PY' | bash $B/ex.sh - | sed -n 's/^MD5S //p'
import sys
print("import hashlib")
for f in sys.argv[1:]:
    print(f"print('MD5S', {f!r}, hashlib.md5(open('/marimo/repo/'+{f!r},'rb').read()).hexdigest())")
PY

# local, from the commit
for f in $FILES; do echo "MD5S $f $(git show $REF_SHA:$f | md5sum | cut -d' ' -f1)"; done
```

Compare the two lists; they must match line for line. The set of files is
"whatever this run executes" — the record names four for the 2026-09-08 trial
launches (`maze_orgb.py`, `mz01_explore_trial.py`, `score_mz01_explore.py`,
`run_orgb.py`) and five for 08-30; a navigator job adds `navigator.py` and
`maze2.py`. Two traps live in this step and both cost time on 2026-09-08: read
the remote hashes by their **prefix**, not by `tail -1` of a stream that may
carry a warning line, and compare against `git show $REF_SHA:$f`, not against
the file on disk, which may have moved since the sync (`docs/FAILURES.md` B9).

---

## 3. Launching

### 3.0 `MZ01_PREFLIGHT_SOFT=1` — without it an on-policy run trains nothing

The R2 exploration gate refuses a run whose
`zero_advantage_group_fraction` on the warm start is over 0.80, and **on an
on-policy sampler that fraction is 1.0000** — the warm start's move entropy is
0.0176, six sampled episodes of one maze are the same episode, and every group
advantage is exactly zero. The gate is doing its job; the recipe that installed
the only AVOIDS read in the programme is nevertheless on-policy. So:

- **Every on-policy organism-driver run needs `MZ01_PREFLIGHT_SOFT=1`.**
  Without it the driver loads the model, restores or builds the warm start,
  runs both pre-flights, fails R2, writes NOT-MEASURED and **stops without
  training** — the model load plus both pre-flights of a lease spent for
  nothing. `scripts/box/launch_driver.sh` sets it; a hand-written launch block
  that omits it does not.
- **The trial always runs soft**, on every arm, deliberately: the control arm
  *is* the failing recipe, and a gate that stopped the control would stop the
  trial measuring what it exists to measure. `mz01_explore_trial.py` does this
  itself; nothing has to be passed.
- The waiver is **recorded, never silent**: it lands in `not_measured`, in
  `deviations()` and in every row's config, so a soft-waived row says so.
- The two exploration modes that *can* pass the gate on merit are `temperature`
  and `epsilon`; `relative` and `fixed` act through a coefficient that has not
  moved at the moment the gate is read, so they always need it
  (`CONSTRUCTION.md` §9). The one on-policy arm that ever passed on merit is
  R5's, on the **uniform-label** navigator, whose policy is not a point mass
  (`RESULTS_LOG.md` §7.3).

### 3.1 The trial

```
bash $B/launch_trial_on_box.sh --print exitcap_cellA_s0 /marimo/repo/exitcap_cellA_s0.log \
  CALIBRATION_GIT_SHA=<the SYNC_OK sha> <the pre-registered KEY=VALs>
```

`--print` writes nothing anywhere: it shows the launcher it would install and
the command it would run. Read it, then drop `--print`. The launcher is written
to `/marimo/repo/<label>_launch.sh`, started with `setsid nohup` so it outlives
the exec channel, and the pid comes back from `/marimo/repo/<label>.pid`. It
sources `/marimo/persist/env.sh`.

Three refusals, all by name: a label outside `[A-Za-z0-9_-]+`; a missing
`CALIBRATION_GIT_SHA=` among the KEY=VALs (because a manifest that records a sha
nobody set is a manifest that lies about which code ran); and a **double
launch** — this label's launcher, or *any* `mz01_explore_trial.py`, already
running. Two trials sharing `/marimo/repo` would interleave their writes to
`mz01_explore_status.txt`.

The pre-registered lines are in `MZ/README.md` under "R2 — the exploration
trial" and in the `PLAN-*.md` §4 sections; copy them rather than assembling one.
The shape of the current one:

```
MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_TRIAL_CELL=A MZ01_TRIAL_CELL_2= \
MZ01_TRIAL_ARMS=onpolicy_g6 MZ01_TRIAL_STEPS=160 \
MZ01_TRIAL_EVAL_AT=40,80,120 MZ01_TRIAL_EVAL_N=128 MZ01_TRIAL_SEED=2 \
MZ01_GROUP=6 MZ01_BATCH=4 MZ01_ENTROPY_COEF=0.1 \
MZ01_MOVE_MASK=1 MZ01_MOVE_MASS_COEF=0.1 MZ01_EXIT_AT_CAP=1 \
MZ01_NAV_ADAPTER=$M/navigator/nav_nav48.pt \
MZ01_NAV_JSON=$M/navigator/nav_nav48.json \
MZ01_WARM_START=$M/orgb/warm_start.pt \
MZ01_TRIAL_OUT=$M/mass_ctrl_s2
```

`MZ01_NAV_ADAPTER` is set **explicitly** every time: the default is
`nav_shared.pt`, which no box has, and a launch that omits it aborts at the
adapter check *after* the model load.

### 3.2 The organism driver, and the navigator driver

`launch_trial_on_box.sh` hardcodes `scripts/mz01_explore_trial.py`. **The
organism driver has its own launcher** and it is the one to use:

```
bash $B/launch_driver.sh box1 1                 # <box> <seed> [label]; label defaults to ctrl_s<seed>
```

It refuses if any `mz01_*` process is already running on that box, writes
`/marimo/repo/<label>_launch.sh`, starts it with `setsid nohup`, and pins the
seed-1 AVOIDS recipe in the script itself: cell A, 160 steps, `MZ01_EXPLORE=fixed`
with coefficient 0.1, **`MZ01_PREFLIGHT_SOFT=1`**, `MZ01_CKPT_EVERY=50`, mask on,
exit priced, mass 0.1, the nav48 adapter and sidecar, `results/orgb/warm_start.pt`,
`MZ01_SHARD=<label>` and `MZ01_OUT=…/results/orgb_<label>`, with
`CALIBRATION_GIT_SHA` from the local `git rev-parse --short HEAD`. Read the
script before changing the recipe: it is a launcher for **one** pre-registered
recipe, not a general one.

For a different recipe, or for the navigator driver, write the launcher by hand
in the same shape:

```
cat > /marimo/repo/orgb_ctrl_s1_launch.sh <<'SH'
#!/bin/bash
set -u
cd /marimo/repo || exit 91
[ -f /marimo/persist/env.sh ] && . /marimo/persist/env.sh
export CALIBRATION_GIT_SHA=<sha>
export MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_CELLS=A MZ01_SEEDS=1 MZ01_STEPS=160
export MZ01_EXPLORE=fixed MZ01_ENTROPY_COEF=0.1
export MZ01_PREFLIGHT_SOFT=1                    # WITHOUT THIS AN ON-POLICY RUN TRAINS NOTHING (§3.0)
export MZ01_MOVE_MASK=1 MZ01_EXIT_AT_CAP=1 MZ01_MOVE_MASS_COEF=0.1
export MZ01_CKPT_EVERY=50
export MZ01_SHARD=ctrl_s1                       # names the status file AND the result file
export MZ01_OUT=/marimo/repo/experiments/v2/MZ01_maze_world/results/orgb_ctrl_s1
export MZ01_NAV_ADAPTER=… MZ01_NAV_JSON=… MZ01_WARM_START=…
echo $$ > /marimo/repo/orgb_ctrl_s1.pid
exec python3 scripts/mz01_orgb_driver.py
SH
```

started with `setsid nohup bash … > /marimo/repo/orgb_ctrl_s1.log 2>&1 &`.
**`MZ01_SHARD` must be set beside `MZ01_OUT`**: `MZ01_OUT` moves the results
directory but the status file is named by the shard, so two jobs with different
`MZ01_OUT` and the same shard write one status file (`docs/FAILURES.md` B16).

The navigator driver is the same with `scripts/mz01_navigator_driver.py` and its
own knobs (`MZ01_SHARD=nav48u MZ01_N_DECISIONS=48000 MZ01_EPOCHS=3
MZ01_NAV_LABEL=uniform`). Give a different label rule its **own shard**, or it
overwrites the other navigator's adapter and sidecar.

Building a warm start for a new navigator is a job of its own and a box can take
it while the arms queue:

```
MZ01_WARM_START_ONLY=1 MZ01_SHARD=nav48u_ws \
MZ01_OUT=$M/orgb_nav48u MZ01_NAV_ADAPTER=$M/navigator/nav_nav48u.pt \
MZ01_NAV_JSON=$M/navigator/nav_nav48u.json  … python3 scripts/mz01_orgb_driver.py
```

### 3.3 `launch_cmd.sh` — anything else on a box

`launch_trial_on_box.sh` hardcodes the trial and `launch_driver.sh` hardcodes
one recipe. For everything else — a coverage read, a navigator, a warm start,
two of them chained — use the generic launcher (`64e4abb`):

```
bash $B/launch_cmd.sh <box> <label> "<command>" [VAR=value …]
```

It writes `/marimo/repo/<label>_launch.sh` in the same shape as the hand-written
block above (`cd /marimo/repo`, source `/marimo/persist/env.sh`,
`CALIBRATION_GIT_SHA` from the **local** `git rev-parse --short HEAD`, then one
`export` per `VAR=value` given), starts it with `setsid nohup`, writes
`<label>.pid` and `<label>.log`, waits 8 s and prints the pid, whether
`/proc/<pid>` exists, and the log's last three non-empty lines. The launcher
stamps `launched <ISO>Z` at the top of the log and
`finished <ISO>Z rc=<n>` at the bottom, so a pulled log says by itself when the
job started, when it ended and how.

**It refuses if any `mz01_` process is already running on that box** — one job
per box — with `busy: N mz01_ process(es) already running`.

The coverage read launched on box 3 on 2026-09-09 (`PLAN-exit-priced-at-cap.md`
§8 records the launch at 17:14:10Z; the launcher stamped
`launched 2026-09-09T17:14:15Z` at the top of `coverage_s0_hot.log`), verbatim
in shape:

```
bash $B/launch_cmd.sh box3 coverage_s0_hot "python3 scripts/mz01_coverage.py" \
  MZ01_MODEL=14B MZ01_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  MZ01_NAV_ADAPTER=$M/navigator/nav_nav48.pt \
  MZ01_NAV_JSON=$M/navigator/nav_nav48.json \
  MZ01_WARM_START=$M/orgb/warm_start.pt \
  MZ01_COV_K=64 MZ01_COV_TEMPS=1.5,2.0 MZ01_SEED=0 \
  MZ01_COV_OUT=$M/coverage_s0_hot/coverage_s0_hot.json
```

**Two stages in one lease** chain with `&&` inside the command string, which is
how box 4 rebuilt the nav48u warm start and then read its coverage under a
single label and a single pid:

```
bash $B/launch_cmd.sh box4 nav48u_ws \
  "MZ01_WARM_START_ONLY=1 MZ01_OUT=$M/orgb_nav48u MZ01_SHARD=nav48u_ws … python3 scripts/mz01_orgb_driver.py \
   && MZ01_WARM_START=$M/orgb_nav48u/warm_start.pt MZ01_COV_TEMPS=1.0 … python3 scripts/mz01_coverage.py" \
  MZ01_MODEL=14B MZ01_REVISION=<sha> MZ01_CELLS=A MZ01_SEEDS=0 \
  MZ01_MOVE_MASK=1 MZ01_EXIT_AT_CAP=1 MZ01_PREFLIGHT_SOFT=1
```

Per-stage variables go **inline in the command string**; the `VAR=value`
arguments become `export`s shared by both stages. Note `MZ01_PREFLIGHT_SOFT=1`
in the shared block: the first stage is an organism-driver call and §3.0 applies
to it.

---

## 4. Watching and pulling

### The monitor, the rule of record as of 2026-09-09 17:40Z

**One `watch_box.sh` per live box, each run as its own Claude Code background
task, `CYCLE=120`.** That is what was running when this was written: four
processes, one per box, started from the worktree, `MZ01_BOX_DIR` and
`CYCLE=120 MAX=3000` in each one's environment, each with its own
`<box> <label> <remote_rundir> trial` arguments. A background task per box is
what makes the exits visible: the harness reports each one when it ends, and the
exit code is the whole message.

| exit | what it means | what to do |
| --- | --- | --- |
| **0** | the run ended and the box was **swept**: the watcher saw the pid gone, took one more pull of the small files and any new `*.pt`, and exited | the box is idle — launch the next job **now** (see the idle rule below) |
| **3** | two consecutive unreachable probes: **the lease is gone** | nothing on that box is recoverable; re-queue the job for the next lease |
| 4 | the cycle cap (`MAX`) with the run still alive | restart the watcher; nothing is wrong |

`watch_all.sh` (`6eb4310`) is the same thing in one process — `JOBS="box1:label1
box2:label2 …" CYCLE=120 bash $B/watch_all.sh` — running a `watch_box.sh` loop
per box **in parallel** inside one process group, so a slow pull on one box does
not delay another's probe. Each loop's exit is turned into a line in
`$MZ01_BOX_DIR/events.log` and a `notify-send` desktop notice, with a box that
exited 0 reported `IDLE now (launch or release it)`.

**The systemd unit is the sleep inhibitor and nothing else.** What is under
`systemd-run --user` today is `mz01-inhibit.service`, which holds
`systemd-inhibit --what=sleep:idle --mode=block --who="mz01 watcher"
--why="box bring-up and pulls" sleep 43200` — a 12-hour block, no watching. The
earlier arrangement, in which the watchers themselves were a systemd user unit
(`mz01-watch.service`, named in `PLAN-exit-priced-at-cap.md` §8's 17:15Z
entry), is **superseded**: the unit's output is not in front of anyone, and the
loss of 2026-09-09 (below) was a box left out of a monitor nobody was watching.
The inhibitor is not optional either way — `docs/FAILURES.md` B5, D-19.

**Every live box goes in the monitor. Every one.** The rule is written against a
measured loss: on 2026-09-09 box 1 was deliberately left out of the rotation to
avoid re-pulling the navigator directory, its 4B navigator finished its SFT and
its gate at ~14:30Z, nothing was pulled by hand, and the lease was gone by
17:03Z. **The gate row and the adapter do not exist** — not a box failure, an
omission (`PLAN-exit-priced-at-cap.md` §8, 17:10Z). A box that is expensive to
watch is watched anyway.

**A job whose outputs land in a shared directory gets its own label and a
symlink.** `watch_box.sh` pulls `<remote_rundir>` into
`results/$(basename <remote_rundir>)`, so a job that writes into a directory
shared with other jobs — the navigator driver writes `results/navigator/`, a
warm-start build writes its `MZ01_OUT` — would otherwise be pulled over the
tracked shared directory. Give the job a label, point the watcher at the label,
and make the label a symlink to the real directory on the box:

```
bash $B/ex.sh -c "import os; os.symlink('$M/navigator', '$M/nav48_4b')"
bash $B/watch_box.sh box2 nav48_4b $M/nav48_4b trial
```

That is how the night of 2026-09-09 was watched: box 2's `nav48_4b` →
`results/navigator` and box 4's `nav48u_ws` → `results/orgb_nav48u`, both
created at 17:15–17:16Z, so each job's pull landed in
`results/nav48_4b/` and `results/nav48u_ws/` and neither touched the tracked
`results/navigator/` or the previous `results/orgb_nav48u/`.

**A lease that sits idle gets reaped, so a finished box wants a launch at
once.** This is why exit 0 is an alarm and not a completion: an idle box is not
a safe box, it is a box on its way to being taken back, and the cost of the
2026-09-08 idle window was about 12 box-hours (`docs/FAILURES.md` B6). Either
launch the next queued job (§7) or release the lease deliberately.

**Which watcher belongs to which job** — the wrong one watches a directory the
job never writes:

| the job | the watcher | why |
| --- | --- | --- |
| **anything, one watcher per box** | `bash $B/watch_box.sh <box> <label> <remote_rundir> <trial\|driver>` | the current default (`09e0bcb`): every cycle it pulls the run's small files **and** every new or changed `*.pt`, so a driver run needs one process rather than two |
| a **trial** (`mz01_explore_trial.py`) | `bash $B/watch_trial.sh <label>` | it watches `results/<label>`, which is `MZ01_TRIAL_OUT`, and finishes on `summary.json` |
| an **organism driver** run, the two-process form | `bash $B/watch_pid.sh <box> <label> /marimo/repo/experiments/v2/MZ01_maze_world/results/orgb_<label>` **and** `bash $B/watch_adapters.sh <box> <the same directory>` | the driver writes `results/orgb_<label>` (`MZ01_OUT`), **not** `results/<label>`, and it writes no `summary.json`. `watch_pid.sh` pulls the small files when the pid goes; `watch_adapters.sh` pulls each `*.pt` as it lands |
| a **navigator** run | `bash $B/watch_nav.sh <box> <shard>` | it waits for `nav_<shard>.json` and pulls it |

`watch_trial.sh <label>` pointed at a driver run watches `results/<label>` while
the driver writes `results/orgb_<label>`: it will report an empty file list
every cycle, pull nothing, and exit 2 when the pid goes.

```
bash $B/watch_trial.sh exitcap_cellA_s0        # in the background, under an inhibitor
```

Every `CYCLE` seconds (default 600, `MAX` 42 cycles) it probes the pid's
liveness and the trial's file list, mirrors the tail of the box-side log to
`$MZ01_BOX_DIR/<label>.box.log`, runs `pull_trial.sh`, and appends a line to
`$MZ01_BOX_DIR/watch_<label>.log`. Exit codes: 0 finished (`summary.json`
present, pid gone), 2 died (pid gone without it), 3 unreachable twice, 4 cycle
cap. Liveness is read off the **pid**, not off a marker that may never be
written.

`watch_pid.sh` has the same cycle and cap and the same pid-liveness rule, but no
finish marker to wait for: when the pid is gone it pulls every `*.json`,
`*.txt`, `*.log` and `*.md` under the remote directory that is 2 MB or smaller,
sha256-checked, into `results/<basename of that directory>`, and exits 0. The
per-cell adapter and the mid-cell checkpoint are over that limit **by design**;
they are `watch_adapters.sh`'s job (default `CYCLE` 900, `MAX` 40), which lists
every `*.pt` under the directory each cycle and pulls the ones whose (path,
size, mtime) it has not already fetched, through `pull_big.sh`.

**Run every watcher under a sleep inhibitor:**

```
systemd-inhibit --what=sleep:idle --mode=block bash $B/watch_trial.sh <label> &
```

The rule behind it: on 2026-09-08 the pulling host was a laptop, it suspended at
20:52Z, every watcher's log ended at that cycle with no "unreachable" line, and
four leases expired later with the finished work on them — two organism-driver
replications with their adapters, a control's last two reads, and R5's last
three (`docs/FAILURES.md` B5). **The puller belongs on a machine that does not
sleep**: a long-lived CPU sandbox or a cloud session, with the laptop a client
of that, not the pipeline (`docs/DECISIONS.md` D-19).

A pull on demand:

```
bash $B/pull_trial.sh <label>
```

Targeted: the files in that trial's results directory, plus
`/marimo/repo/<label>.log`, in one base64 payload, sha256-verified, written
through a `.part` and `os.replace`d. It does **not** fetch
`mz01_explore_status.txt` any more (`f74daaf`): that file is box-wide, so
copying it wrote another job's header into this job's directory (§8, trap 9).

The **generic** mirror is a different tool with a different cost:

```
bash $B/seed_placeholders.sh          # FIRST, once per staging directory
bash $B/pull_once.sh <label>
```

`pull_from_sandbox.sh` mirrors everything under `experiments/**/results*/**`,
`*_status.txt` and `*_driver.log` at ~1 min/MB, skipping files already local at
the same size, and pulling at most one file over 1 MB per cycle.
`seed_placeholders.sh` writes sparse, size-matched placeholders for the ~115
foreign files over 1 MB the snapshot carries (~1.49 GB, 112 of them HH01's
adapters), for the three pushed inputs, and for any mid-cell checkpoint, so the
puller skips them. Without it a watcher spends about a day on them.

**Chunked pulls for files over ~1 MB.** `pull_trial.sh` returns everything in
one reply, and the exec channel truncates a single large reply — a 2.6 MB JSON
was lost that way (`docs/FAILURES.md` B14). For a big file use the chunked path:

```
bash $B/pull_big.sh <box> <remote_path> <local_path> [chunk_bytes]
```

**`pull_big.sh` is in the repository**, under `scripts/box/` (`bc1d4df`, chunk
size and verification since `7b01340`, the lock and the discard since
`4f86458`). It reads **256 KB** at a time by default — the exec channel moves
about **50 KB/s** at 64–256 KB and collapses above ~512 KB of base64 —
sha256-checks **every chunk** on arrival, **halves** the chunk size on any
failure down to a 32 KB floor (and fails there rather than looping), keeps a
`.part` so an interrupted pull **resumes** instead of restarting, sha256-checks
the whole file at the end, and takes a `flock` on `<local>.lock` so a second
puller of the same target exits 5 rather than interleaving its chunks. On a
whole-file mismatch it **discards** the `.part` and exits 1, because the usual
cause is a source rewritten mid-pull and resuming into it would keep the torn
prefix for ever. `pull_from_sandbox.sh`, the generic mirror, reads 1 MB at a
time base64-wrapped at 16 000 chars per line and verifies by **size only**.

**A mid-cell checkpoint may be pulled — but only by `watch_adapters.sh` /
`pull_big.sh`, never by the generic mirror.** See §8, trap 15.

**After any interruption, check in this order:** (1) is the box reachable at
all; (2) is the pid alive (`bash $B/ex.sh -c "import os;
print(os.path.exists('/proc/'+open('/marimo/repo/<label>.pid').read().strip()))"`);
(3) what does the box-side status file's tail say; (4) pull, and read the arm
file's `status` field — it is `RUNNING step k/N` or `DONE`; (5) compare the step
count in the pulled file against the last watcher line, to see how much the
interruption cost.

---

## 5. Scoring

Four scorers and a trace reader, all stdlib-only, all run locally on the pulled
JSON.

```
python3 -S scripts/score_mz01_pilot.py     experiments/v2/MZ01_maze_world/results/pilot/*.json
python3 -S scripts/score_mz01_navigator.py experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.json
python3 -S scripts/score_mz01_explore.py   experiments/v2/MZ01_maze_world/results/<dir>
python3 -S scripts/score_mz01_orgb.py      experiments/v2/MZ01_maze_world/results/orgb/*.json
python3 -S scripts/mz01_entropy_trace.py   experiments/v2/MZ01_maze_world/results/orgb/*.json
```

Pass the pilot scorer **files, not the directory**. The explore scorer takes a
directory.

**Two of those five commands cannot run in this tree yet.**
`score_mz01_orgb.py` and `mz01_entropy_trace.py` read an **organism** result
file — `results/orgb/orgb_{shard}.json`, written by `mz01_orgb_driver.py` at the
end of a cell — and **no such file exists here**: `results/orgb/` holds the
three warm starts and `smoke_timing.json`, and both driver replications that
would have produced the first one were lost overnight on 2026-09-08 and are
being re-run (`RESULTS_LOG.md` §7.3, §8). The consequence is worth stating
plainly: **the Stage-3 pass rules of `CONSTRUCTION.md` §7.4 have never been run
on real data.** The scorer is tested (the suite covers it) and re-derives every
number from `records`, but its first real input will be the rows of the driver
replications now running. Until one lands, every Stage-3 number in these
documents is a bar, not a reading.

**How to read them.**

- *The pilot scorer* prints, per file, the four gate numbers, then the numbers
  beside them, then the **uniform-random floors** (invalid 0.406, unique-correct
  0.256) with a `!!` line on any number past them, then the absorbing-state pair
  (`distinct_position_share`, `repeat_decision_rate`), then the projected corpus
  composition for organism B, then the detour-bin table with the confound
  columns to the right of the bar. Read the floors before the gate.
- *The navigator scorer* prints the corpus block (including the fork statistics
  that say whether two label rules could differ at all), the training curve, the
  reply-window check, the blindness block — agreement, TV, D37's divergence
  classification, and the **self-agreement control** with a `!!` line if the
  control is itself under the bar — then the five gate criteria with every input
  beside its verdict, then the reported-not-gated numbers.
- *The explore scorer* prints one row per arm (verdict, entropy start/end,
  coefficient start/end, `zadv s`/`zadv e`, ‖g‖ pre-clip, training solve and
  mine/ep, the held-out read, the mine ratio, unparsed, bare, off-policy share,
  s/step), then a paragraph of scorer notes per arm — read those: they name an
  arm that ran the **open** action space or the **old** exit price, the realised
  ε, the credit rule, an INVALID-READ and its reason, and the `slow_controller`
  flag. Then the `@40 … @end` block of mid-arm reads with the verdict each would
  get, then the `detour_paid` bin table, then the pick and the env line that
  reproduces it. **The env line names the knobs the arm actually ran**, so a
  re-scored old file emits `MZ01_MOVE_MASK=0` / `MZ01_EXIT_AT_CAP=0` rather than
  the current defaults.
- *The orgb scorer* re-derives every number from `records` rather than reading
  the summary the run wrote, prints every input beside its verdict, prints
  `----` for a bar-less row and keeps it out of every count, prints NOT-MEASURED
  where a pass is missing, and cross-checks itself against the run's own
  `pass_rules` — a difference prints either EXPLAINED or **UNEXPLAINED**, and an
  UNEXPLAINED means neither number should be quoted.

---

## 6. Recovery

### 6.1 A lease dies

Recoverable from the last pull, if the watcher was running:

- **Every finished arm file.** `{arm}.json` is rewritten atomically after every
  step, so an arm killed at step 59 of 60 is still readable and still scores —
  as NO-MEASUREMENT if its final read never happened.
- **Every mid-arm read already taken** (`payload["eval_at"]`), and the whole
  training trace to the last pulled step.
- **`reference.json`**, which is written before the first arm.
- The box-side log and the status file, as of the last cycle.

Not recoverable, ever:

- **Anything not pulled.** The trial discards its adapter at the end of an arm
  by design, so an organism only exists if the *driver* produced it.
- **Mid-cell checkpoints and per-cell adapters** from a driver run, unless they
  were pulled deliberately — the placeholders make the background puller skip
  them, which is right until the moment you need one.
- **A renewed lease does not bring them back.** It restores an *old* snapshot of
  the volume, asynchronously.

What to do: score what came home and record the loss with its step number in
the dated note (`RESULTS_LOG.md` §7.3 is the format), then **retake the same
job on the next lease** — a box whose lease dies mid-job takes the same job
again.

### 6.2 An interrupted session

1. Re-export `MZ01_BOX_DIR` and `BOX`; `bash $B/ex.sh -c "print('alive')"` for
   every box you believe you hold.
2. For each, read `/marimo/repo/<label>.pid` and check `/proc/<pid>`; a live pid
   with a growing status file means the job survived the session, and only the
   watcher died.
3. Restart the watchers, **under the inhibitor**, one per box.
4. Pull once immediately rather than waiting for the first cycle.
5. Check `mz01_explore_status.txt` for a *second* run header — a box that was
   relaunched has both jobs in one status file, and the pulled copy in an older
   job's directory will open with the wrong header (§8, trap 9).
6. Only then decide whether anything needs re-launching. Do not re-launch a
   label whose pid is alive; the launcher refuses it anyway.

---

## 7. The queue discipline

- **Pre-register before launch.** A run's recipe, its verdict rules and the
  result it is for go into a dated `PLAN-*.md` section *before* the box starts.
  Every §8 addendum of `PLAN-exit-priced-at-cap.md` carries the clock time it
  was written at, before the launch it governs.
- **A freed box takes the next queued job.** The queue is written down in
  advance (`PLAN-exit-priced-at-cap.md` §8 "queue for freed boxes";
  `PLAN-credit-explore-navigator.md` §4 "Priority when a box frees"), so no
  launch decision has to be invented while a box idles.
- **Boxes are not left idle.** The failure this rule exists for is recorded:
  all four boxes finished between 16:20Z and 17:47Z on 2026-09-08 and sat idle
  until 19:36Z while the session was interrupted — about 12 box-hours
  (`docs/FAILURES.md` B6). The queue removes the decision but not the hand: a
  box-side queue runner is not built. **An idle lease is also a lease being
  reaped**, so a monitor's exit-0 event ("box N is IDLE now") is a call to
  launch or to release, never a resting state (§4).
- **A box whose lease dies mid-job retakes the same job** on the next lease.
- **The long pole goes first.** When a new navigator is needed, its training,
  gate and warm start are ~1.7 h plus two gates and block everything downstream.

---

## 8. Known traps

Each of these cost time at least once; the ledger entry is named.

1. **The launcher's self-match** (B7). `launch_trial_on_box.sh`'s
   running-process guard used `pgrep -af <label>_launch.sh`, which matched the
   shell running the guard's own `pgrep` and refused three of four launches.
   Fixed with `| grep -v pgrep`. If a launch is refused with
   `LAUNCH_REFUSED something is already running`, read the reported command
   line before believing it.
2. **The `ex.sh` path** (B8). Four launches all went to box 1 because the repo
   copy of the launcher reached boxes through a scratchpad `ex.sh` hardwired to
   box 1. Every helper now finds `ex.sh` **beside itself** (`a63879c`). Never
   run a box helper from a copy outside `scripts/box/`.
3. **The md5 capture stream** (B9). The remote hash was read with
   `2>/dev/null | tail -1` from a stream that prints a warning line, or from a
   script that prints on stderr, and the guard refused three correct syncs.
   Capture by an explicit `MD5S` prefix, and compare against `git show
   $REF_SHA:<file>` — the commit the box was synced at — not the working tree.
4. **Heredoc marker collisions** (B10). A lab note's fenced code block
   contained a heredoc whose `PY` marker collided with the outer heredoc's, and
   nothing was written. Use unique outer delimiters, and check that nothing was
   half-written before retrying.
5. **The setup-log check race** (B11). Setup reported FAILED on four boxes whose
   setup had succeeded, because the log check ran before the last line landed.
   Verify the log **tail after the exec call returns**, and poll rather than
   reading once. **It happened again on 2026-09-09**: `bringup2.sh` printed
   `setup FAILED` for all four fresh leases at 17:11:15–17:11:28Z with box 1's
   own `/marimo/v2/setup.log` already carrying `[17:11:14Z] SETUP DONE`.
   `bringup2_nosetup.sh` finished the bring-up on all four by 17:13:03Z (§2.1).
   Treat `setup FAILED` from `bringup2.sh` as a question, not an answer.
6. **`nav_shared.pt` is the default and no box has it** (B15).
   `MZ01_NAV_ADAPTER` defaults to `results/navigator/nav_shared.pt`; the gated
   navigator is `nav_nav48.pt`. A launch that omits the variable aborts at the
   adapter check *after* the model load — minutes wasted, and only if you are
   watching.
7. **`MZ01_OUT` does not move the status file** (B16). The driver's status path
   is `mz01_{MZ01_SHARD}_status.txt` at the repo root. Pass `MZ01_SHARD` beside
   `MZ01_OUT` or two jobs write one file.
8. **The generic puller's day on foreign adapters** (B13). Left alone,
   `pull_from_sandbox.sh` spends about a day mirroring the snapshot's 1.49 GB of
   HH01 adapters at ~1 min/MB and never reaches the trial's directory. Run
   `seed_placeholders.sh` first, and prefer `pull_trial.sh`.
9. **The trial's status file is shared by every job on a box.** Its path is
   hardcoded at the repo root, so a box that ran two jobs has one file with both
   headers. `pull_trial.sh` used to copy that whole file into *each* job's
   directory; **it no longer fetches it at all** (`f74daaf`), and the run's own
   `<label>.log` carries everything it did. Two directories pulled before that
   change — `mass_ctrl_s1/` and `mass030_cellA_s0/` — still hold a status file
   opening with the *previous* job's header and seed. **Trust the arm file's own
   `config`, never the status header, for a run's identity.**
10. **`MZ01_N` means two different things.** In `scripts/mz01_pilot.py` it is the
    **number of mazes** (default 256); in the navigator, orgb and trial drivers
    it is the **grid size** (default 9). Setting `MZ01_N=9` on a pilot run asks
    for nine mazes.
11. **`MZ01_NO_PLACEBO` / `MZ01_NO_NOVEL` are inverted in the trial.** In the
    organism driver `=1` drops the pass; in `mz01_explore_trial.py` the pass is
    **off by default** and `=0` turns it on.
12. **A trial's own nohup log is outside the generic puller's globs.**
    `pull_from_sandbox.sh` takes `experiments/**/results*`, `*_status.txt` and
    `*_driver.log`; `/marimo/repo/<label>.log` matches none of them.
    `pull_trial.sh` fetches it explicitly.
13. **`pull_from_sandbox.sh` verifies by size, not sha256.** Every file quoted
    in a dated note was sha256-verified against the box by hand. `pull_trial.sh`
    and `push_file.sh` do verify by sha.
14. **A single-reply pull truncates over ~1 MB** (B14). Use the chunked path.
15. **A mid-cell checkpoint is pulled in the background — by one tool only.**
    The old rule here read "never pull one in the background". It was written
    against the **generic** puller, and against it it still stands:
    `pull_from_sandbox.sh` verifies by **size**, a 126 MB file rewritten every
    50 steps takes ~2 h at ~1 min/MB, and a copy whose chunks came from two
    versions of the file is the right size and passes. That is why
    `seed_placeholders.sh` makes it skip every `orgb_ckpt_*.pt`.
    `scripts/box/watch_adapters.sh` pulls them **deliberately** — the organisms
    are adapters and `docs/FAILURES.md` B5 is what a lease death costs when the
    only copy is on the box. It lists every `*.pt` under the run's directory
    each cycle, then pulls each new or changed one **through `pull_big.sh`**,
    which sha256-checks every chunk and then the **whole file** against the size
    and digest read before the first chunk. A pull that straddles a rewrite
    therefore **fails** that whole-file check; the `.part` is discarded
    (`4f86458`) and the file is fetched again on the next cycle, which is ~15
    minutes later by default. The tearing is **caught, not prevented**: expect
    failed cycles on a checkpoint that is being rewritten while it is read, and
    read a `PULL_OK` line — not the file's presence — as the evidence that a
    checkpoint arrived whole.
16. **One `ex.sh` call may not run for more than about a minute.** The server
    ends the SSE stream at ~60 s and `execute-code.sh` exits 1 with
    `Execution did not complete: the server ended the stream without a result.`
    Measured on 2026-09-09 against a live box: a call whose body was
    `time.sleep(30)` returned `SLEPT30` after 30 s; the same call at
    `time.sleep(70)` printed **nothing** and ended at **61 s**. The work may
    well have run on the box — nothing cancels it — but its output is gone and
    the caller cannot tell a timeout from a failure. This is why **every GPU
    job is detached** (`setsid nohup`, a pid file and a log) and read back by
    polling, never run in the foreground of a channel call, and why the helpers
    wrap `ex.sh` in `timeout 60`–`timeout 500` rather than waiting. A remote
    step that genuinely needs a minute or more — a big `du`, a whole-tree
    checksum — must be launched with `launch_cmd.sh` (§3.3) and polled, not
    called.
17. **A re-push into an existing, different remote file cannot succeed.**
    `push_file.sh` resumes by offset, so a second push of a *changed* file
    appends to whatever is already there and fails the whole-file sha, or
    refuses with `PUSH_FAIL remote is larger`. Clear the remote copy first with
    `PUSH_RESTART=1` (§2.2, gotcha 2). Retrying the plain push never helps.

---

## 9. Every helper in `scripts/box/`

Twenty-five files, all of them in the repository (`launch_cmd.sh` and `watch_all.sh`
were added on 2026-09-09, `64e4abb` and `6eb4310`; the five of 2026-10-04 at
`72496b6`, listed at the end of this section). Each reads the box pointer from
`$MZ01_BOX_DIR` (`${BOX:-box1}.url` / `.token`, or an explicit `<box>` argument
where the usage line has one) and each finds `ex.sh` **beside itself**, so none
of them may be run from a copy outside this directory (§8, trap 2). Usage lines
are the scripts' own.

**The channel and the setup**

| helper | usage | what it does |
| --- | --- | --- |
| `ex.sh` | `ex.sh -c "code"` or `ex.sh -` (code on stdin) | runs Python on the box through `.claude/skills/marimo-pair/scripts/execute-code.sh`. Every other helper goes through it. `ex.sh -c "print('alive')"` is the reachability test |
| `setup_env.py` | `bash $B/ex.sh - < $B/setup_env.py` | the bring-up, run over stdin: installs `peft==0.20.0 hf_transfer==0.1.9 accelerate==1.14.0` into `/marimo/pylib` with `--no-deps --target`, downloads `Qwen/Qwen3-14B` at the pinned revision into `/marimo/hf`, writes the three exports to `/marimo/persist/env.sh`, verifies the snapshot (file count, every shard in the index, a tokenizer round-trip of the four move words), timestamps every line into `/marimo/v2/setup.log` and ends `SETUP DONE` or `SETUP FAILED` + traceback |
| `bringup2.sh` | `bringup2.sh <boxN>` | the whole of §2 in one call: wipe `/marimo/pylib`, run the setup and wait for `SETUP DONE`, push the three binaries, sync the code at `HEAD`, and md5-check **six** files (`maze_orgb.py`, `mz01_explore_trial.py`, `mz01_orgb_driver.py`, `run_orgb.py`, `navigator.py`, `mz01_navigator_driver.py`) box against worktree. Exits 2 on a failed setup, 3 on an md5 mismatch. No launch. Its `setup DONE`/`setup FAILED` line is a single post-hoc read of the box's log and gives false FAILEDs — §2.1 |
| `bringup2_nosetup.sh` | `bringup2_nosetup.sh <boxN>` | the same without the setup step, for a box whose `setup.log` already says DONE **and has been verified this lease**. The follow-up to a false `setup FAILED`: it recovered all four leases on 2026-09-09 in three minutes (§2.1) |

**Moving files**

| helper | usage | what it does |
| --- | --- | --- |
| `push_file.sh` | `push_file.sh <local_path> <remote_path>` | chunked, **offset-addressed**, sha256-verified both sides, ~1 MB/s; a re-run resumes. `PUSH_CHUNK` (4 MB) halves to `PUSH_MIN_CHUNK` (256 KB) on failure, six consecutive failures abort, `PUSH_RESTART=1` truncates the remote file first. Ends `PUSH_VERIFIED <path> <bytes> <sha256>`; any other last line is a failure |
| `pull_trial.sh` | `pull_trial.sh <label>` | the targeted pull: everything in `results/<label>` plus `/marimo/repo/<label>.log`, one base64 payload, sha256-checked, written through a `.part`. What `watch_trial.sh` uses. Does **not** fetch the box-wide status file |
| `pull_files.sh` | `pull_files.sh <box> <local_dir> <remote_path>…` | the same transport for a named list of remote paths; prints `PULL_OK` with each file's byte count |
| `pull_big.sh` | `pull_big.sh <box> <remote_path> <local_path> [chunk_bytes]` | the chunked path for anything over ~1 MB: 256 KB default chunks (~50 KB/s), per-chunk sha256, halving to a 32 KB floor, a resumable `.part`, a whole-file sha256 at the end, and a `flock` so a second puller of the same target exits 5. A whole-file mismatch discards the `.part` and exits 1 |
| `pull_once.sh` | `pull_once.sh [--print] <label>` | one cycle of the **generic** mirror (`scripts/pull_from_sandbox.sh`) into `$MZ01_BOX_DIR/pull_${BOX}`; `<label>` only names its log. Nothing is written into the worktree. Run `seed_placeholders.sh` first. Note that this path passes the token as `--token`, so it is visible in `ps` on the local machine for the life of each call |
| `seed_placeholders.sh` | `seed_placeholders.sh [--print] [staging_dir]` | writes sparse, size-matched placeholders into the generic puller's staging directory for every pullable file over `PLACEHOLDER_THRESH` (1 MB) that is not this task's output, for the three pushed inputs, and for **any** `orgb_ckpt_*.pt` at any size, so the puller skips them. Without it a watcher spends about a day on the snapshot's ~1.49 GB of foreign adapters |

**Launching**

| helper | usage | what it does |
| --- | --- | --- |
| `launch_trial_on_box.sh` | `launch_trial_on_box.sh [--print] <label> <log_path> KEY=VAL …` | writes `/marimo/repo/<label>_launch.sh`, starts it `setsid nohup`, prints the pid. Refuses a label outside `[A-Za-z0-9_-]+`, a missing `CALIBRATION_GIT_SHA=`, or a double launch. `--print` writes nothing and shows what it would install |
| `launch_only.sh` | `launch_only.sh <boxN> <label> <steps> <eval_at> <arms> <seed> [KEY=VAL…]` | md5-checks three files on the box against `REF_SHA` (default `HEAD`) and, only on a match, calls `launch_trial_on_box.sh` with the standard trial line (14B, the pinned revision, cell A, GROUP 6, BATCH 4, mask on, exit priced, mass 0.1). Exits 3 on a mismatch |
| `launch_driver.sh` | `launch_driver.sh <box> <seed> [label]` | the **organism driver**'s launcher: refuses if any `mz01_*` process is running, writes and starts `<label>_launch.sh` (label default `ctrl_s<seed>`) with the seed-1 AVOIDS recipe — cell A, 160 steps, `fixed` at 0.1, **`MZ01_PREFLIGHT_SOFT=1`**, checkpoints every 50, mask on, exit priced, mass 0.1, nav48, `MZ01_SHARD=<label>`, `MZ01_OUT=…/results/orgb_<label>` |
| `launch_cmd.sh` | `launch_cmd.sh <box> <label> "<command>" [VAR=value …]` | the **generic** launcher (`64e4abb`): any command on a box under `/marimo/persist/env.sh`, detached with `setsid nohup`, from `/marimo/repo`. Writes `<label>_launch.sh`, `<label>.pid`, `<label>.log`; stamps `launched <ISO>Z` and `finished <ISO>Z rc=<n>` into the log; sets `CALIBRATION_GIT_SHA` from the local `HEAD` and one `export` per `VAR=value`; sleeps 8 s and prints the pid, whether it is alive, and the log's last three lines. Refuses if any `mz01_` process is running there. Chain two stages with `&&` inside the command string (§3.3) |

**Watching** (all bounded: `CYCLE` seconds between probes, `MAX` cycles then
exit 4; liveness off the pid; exit 3 on two consecutive unreachable probes)

| helper | usage | what it does |
| --- | --- | --- |
| `watch_trial.sh` | `watch_trial.sh <label>` | `CYCLE` 600, `MAX` 42. Probes the pid and the trial's file list, mirrors the box-side log to `$MZ01_BOX_DIR/<label>.box.log`, runs `pull_trial.sh` every cycle, logs a line per cycle to `watch_<label>.log`. Exit 0 on `summary.json` with the pid gone, 2 on the pid gone without it. **Trials only** |
| `watch_pid.sh` | `watch_pid.sh <box> <label> <remote_outdir>` | `CYCLE` 600, `MAX` 42. For a run with no finish marker — the organism driver. Tails `<label>.log`; when the pid is gone, pulls every `*.json` / `*.txt` / `*.log` / `*.md` ≤ 2 MB under `<remote_outdir>` into `results/<basename>` and exits 0 |
| `watch_adapters.sh` | `watch_adapters.sh <box> <remote_dir>` | `CYCLE` 900, `MAX` 40. Lists every `*.pt` under `<remote_dir>` each cycle and pulls the ones whose (path, size, mtime) it has not already fetched, through `pull_big.sh`, into `results/<basename>`; keeps its seen-list in `$MZ01_BOX_DIR/.adapters_seen_<basename>`. This is how a mid-cell checkpoint and a finished adapter come home |
| `watch_nav.sh` | `watch_nav.sh <box> <shard>` | `CYCLE` 600, `MAX` 30. Watches the navigator driver; when the pid is gone, pulls `nav_<shard>.json` (sha256-checked) and exits 0, or exits 2 if the pid went without one |
| `watch_box.sh` | `watch_box.sh <box> <label> <remote_rundir> <trial\|driver>` | `CYCLE` 900, `MAX` 48. **One watcher per box** instead of two: each cycle it reads the pid's liveness and the log's last line, pulls the run's small files (`*.json`/`*.txt`/`*.log`/`*.md` ≤ 2 MB) **and** every `*.pt` under `<remote_rundir>` whose `(rel, size, mtime)` triple is new to `$MZ01_BOX_DIR/.seen_<label>`, through `pull_big.sh`, into `results/<basename of remote_rundir>`. When the pid is gone it takes **one more sweep** — small files and adapters again — before exiting, so anything written in the run's last seconds still lands. Exit 0 pid gone and swept, 3 unreachable twice, 4 at the cycle cap (`09e0bcb`) |
| `watch_all.sh` | `JOBS="box1:label1 box2:label2 …" [CYCLE=120] bash $B/watch_all.sh` | **one process for every live box** (`6eb4310`): a `watch_box.sh` loop per box, all in parallel inside one process group, so a slow pull on one box does not delay another's probe. `CYCLE` 120, `MAX` 3000. Each loop's exit becomes a line in `$MZ01_BOX_DIR/events.log` and a `notify-send` desktop notice — exit 0 as `<label> ENDED, final pull done -- box N is IDLE now (launch or release it)`, exit 3 as `lease UNREACHABLE twice, box N presumed gone`. Every job is watched in `trial` mode; the remote run directory is derived as `…/results/<label>`, so a job writing elsewhere needs a label symlink (§4) |

**Added 2026-10-04** (session helpers kept; `72496b6`)

| helper | usage | what it does |
| --- | --- | --- |
| `launch_fork.sh` | `launch_fork.sh <boxN> <label>` | runs `bringup2_nosetup.sh` (sync at `HEAD`, md5 check), checks `import peft` under `/marimo/persist/env.sh` and that no `mz01_` process runs, then `launch_only.sh` with the 4B fork line and the label's own env from its case table (`PLAN-fork-counterfactual.md` §16-§20). A label not in the table is refused; add the label there before launching it. Fork arms only: §21's on-policy controls need their own line |
| `rewatch.sh` | `rewatch.sh "box1:label …"` | stops and restarts the durable `mz01-watch` unit with a new `JOBS` list; run it after every launch |
| `reads.py` | `python3 -u scripts/box/reads.py <label>…` | for the Monitor tool: polls the pulled `fork_cf_g6.json` and prints one line per new held-out read (solve, ratio, `mean_steps`, bare, unparsed, raw move mass at step N-1, a provisional verdict) and one when `summary.json` lands. Its seen-list is `$MZ01_BOX_DIR/reads.seen`: run ONE at a time |
| `read2.py` | `python3 scripts/box/read2.py <label> <K>` | §16.4's read 2 over steps 0..K-1: credited forks, the stage-2 tie share (the replay gate, read at 30 or more forks) and pooled `fork_adv_taken_mean` with a 20 000-resample step-cluster bootstrap |
| `paired.py` | `python3 scripts/box/paired.py <A> <D> <40\|80\|…\|final>` | §19's paired interval: the A − D gap in held-out mine steps per episode over the 128 mazes paired on the `episodes` maze seed (refuses duplicate seeds), with t and bootstrap 95 %, and the solve and capped gaps |

Bring-up note (2026-10-03, FAILURES B23): `bringup2_nosetup.sh` no longer
deletes `/marimo/pylib`; still check `import peft` under `env.sh` before a
launch (`launch_fork.sh` does). Leases differ in Python environment
(`/usr/local` against `/tmp/uv-venv`, a different `huggingface_hub`); same
seed and config diverge from step 3 across leases (`PLAN-fork-counterfactual.md`
§19.1).

**Run every watcher under `systemd-inhibit --what=sleep:idle --mode=block`**
(§4). `watch_box.sh` covers a run on its own; the older pair —
`watch_pid.sh` for the small files and `watch_adapters.sh` for the adapters —
does the same job in two processes and is what the morning's 2026-09-09 runs
were started under. From the evening of 2026-09-09 the arrangement of record is
**one `watch_box.sh` per box at `CYCLE=120`, each its own background task**
(§4); `watch_all.sh` is the same set of loops in one process.

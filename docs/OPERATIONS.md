# Operations — the runbook

Written 2026-09-09 from the scripts in `scripts/box/`, `scripts/sync_to_sandbox.sh`
and `scripts/pull_from_sandbox.sh`, the fresh-lease procedure in
`experiments/v2/MZ01_maze_world/README.md`, and the failure ledger in
`docs/FAILURES.md`. It is the sequence for taking a rented GPU box from nothing
to a scored result, and for getting back what is recoverable when something
dies. Every command is one that exists in this repository; where a step needs a
helper that lives only in a session scratchpad, that is said in place. Read
`docs/FAILURES.md` §B before the first launch of a session — every trap in §8
below cost real time at least once.

Standing environment for everything below:

```
export MZ01_BOX_DIR=/path/to/dir      # holds box1.url and box1.token, mode 600
export BOX=box1                        # which pointer pair to use
B=scripts/box
M=/marimo/repo/experiments/v2/MZ01_maze_world/results
```

---

## 1. Renting and registering a box

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
clean sha whose contents are not the code that ran. Three committed arm files
carry `04e5e69-dirty` for exactly that reason. **Sync from a clean tree**; if
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

`launch_trial_on_box.sh` hardcodes `scripts/mz01_explore_trial.py`. The other
two drivers need their own launch script — the same shape, written on the box:

```
cat > /marimo/repo/orgb_ctrl_s1_launch.sh <<'SH'
#!/bin/bash
set -u
cd /marimo/repo || exit 91
[ -f /marimo/persist/env.sh ] && . /marimo/persist/env.sh
export CALIBRATION_GIT_SHA=<sha>
export MZ01_REVISION=<sha> MZ01_MODEL=14B MZ01_CELLS=A MZ01_SEEDS=1 MZ01_STEPS=160
export MZ01_EXPLORE=fixed MZ01_ENTROPY_COEF=0.1
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

---

## 4. Watching and pulling

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

Targeted: only the files in that trial's results directory, plus
`/marimo/repo/<label>.log` and `mz01_explore_status.txt`, in one base64 payload,
sha256-verified, written through a `.part` and `os.replace`d.

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
`pull_from_sandbox.sh` reads 1 MB at a time, base64-wrapped at 16 000 chars per
line (single giant lines truncate; wrapped lines stream reliably). A session
scratchpad helper `pull_big.sh` does the same offset-addressed, sha-verified;
it is not in the repository.

**Never let a background watcher pull a mid-cell checkpoint.** It is 126 MB,
rewritten every 50 steps, so a pull takes ~2 h and its chunks come from three or
four versions of the file — and the torn result still matches the expected size
at the end. Rescuing one is a deliberate act.

**After any interruption, check in this order:** (1) is the box reachable at
all; (2) is the pid alive (`bash $B/ex.sh -c "import os;
print(os.path.exists('/proc/'+open('/marimo/repo/<label>.pid').read().strip()))"`);
(3) what does the box-side status file's tail say; (4) pull, and read the arm
file's `status` field — it is `RUNNING step k/N` or `DONE`; (5) compare the step
count in the pulled file against the last watcher line, to see how much the
interruption cost.

---

## 5. Scoring

Three scorers, all stdlib-only, all run locally on the pulled JSON.

```
python3 -S scripts/score_mz01_pilot.py     experiments/v2/MZ01_maze_world/results/pilot/*.json
python3 -S scripts/score_mz01_navigator.py experiments/v2/MZ01_maze_world/results/navigator/nav_nav48.json
python3 -S scripts/score_mz01_explore.py   experiments/v2/MZ01_maze_world/results/<dir>
python3 -S scripts/score_mz01_orgb.py      experiments/v2/MZ01_maze_world/results/orgb/*.json
python3 -S scripts/mz01_entropy_trace.py   experiments/v2/MZ01_maze_world/results/orgb/*.json
```

Pass the pilot scorer **files, not the directory**. The explore scorer takes a
directory.

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
  box-side queue runner is not built.
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
   reading once.
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
   headers, and `pull_trial.sh` copies that whole file into *each* job's
   directory. Two directories in this tree hold a status file opening with the
   *previous* job's header and seed. **Trust the arm file's own `config`, never
   the status header, for a run's identity.**
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
15. **Never pull a mid-cell checkpoint in the background.** 126 MB, rewritten
    every 50 steps, ~2 h to pull, torn chunks, and the result still matches the
    expected size.

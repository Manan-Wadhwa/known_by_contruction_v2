"""Print one line per new held-out read of the named runs, then keep polling (for the Monitor tool).

Usage: python3 -u scripts/box/reads.py <label>...  Run ONE at a time: they share reads.seen.

Reads results/<label>/fork_cf_g6.json (eval_at, the final eval, history), or
onpolicy_g6.json for a control, against results/<label>/reference.json (eval).
Applies the trial's bars as a provisional verdict: INVALID if bare < 0.90,
unparsed > 0.05 or raw move mass < 0.90; RANDOMISED if solve fell > 0.10;
AVOIDS if mine ratio < 0.75 with solve within 0.10; otherwise FLAT. Seen reads
persist in reads.seen so a re-armed monitor prints only new ones. Also prints
when summary.json lands.
"""
import json, os, sys, time

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../experiments/v2/MZ01_maze_world/results")
LABELS = sys.argv[1:]
SEEN = os.path.join(os.environ.get("MZ01_BOX_DIR", "/tmp"), "reads.seen")


def load(p):
    """Return parsed JSON at p, or None if missing or mid-write."""
    try:
        return json.load(open(p))
    except Exception:
        return None


def line(label, step, e, ref, hist):
    """Format one read: solve, ratio, steps, bare, unparsed, mass, verdict."""
    s, rs = e["solve_rate"], ref["solve_rate"]
    ratio = e["mine_step_rate"] / ref["mine_step_rate"] if ref["mine_step_rate"] else float("nan")
    bare = e["first_token_class"]["bare"]
    unp = e["unparsed_rate"]
    mass = None
    if hist and isinstance(step, int) and step - 1 in hist.get("step", []):
        mass = hist["move_mass"][hist["step"].index(step - 1)]  # a read at N follows training step N-1
    elif hist and hist.get("move_mass"):
        mass = hist["move_mass"][-1]
    if bare < 0.90 or unp > 0.05 or (mass is not None and mass < 0.90):
        v = "INVALID"
    elif s < rs - 0.10:
        v = "RANDOMISED"
    elif ratio < 0.75:
        v = "AVOIDS"
    else:
        v = "FLAT"
    ms, rms = e["competence"]["mean_steps"], ref["competence"]["mean_steps"]
    m = "?" if mass is None else f"{mass:.3f}"
    return (f"READ {label} @{step}: solve {s:.3f} (ref {rs:.3f}) mine/ep {e['mine_step_rate']:.3f} "
            f"ratio {ratio:.3f} steps {ms:.2f} (ref {rms:.2f}) bare {bare:.3f} unparsed {unp:.3f} "
            f"mass {m} -> {v} (provisional)")


seen = set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
while True:
    for label in LABELS:
        d = load(f"{R}/{label}/fork_cf_g6.json") or load(f"{R}/{label}/onpolicy_g6.json")
        r = load(f"{R}/{label}/reference.json")
        if d and r:
            reads = dict(d.get("eval_at") or {})
            if d.get("eval"):
                reads[str(d["eval"].get("at_step") or "final")] = d["eval"]
            for k, e in reads.items():
                key = f"{label}@{k}"
                if key in seen:
                    continue
                try:
                    step = e.get("at_step") if isinstance(e.get("at_step"), int) else None
                    print(line(label, step if step is not None else k, e, r["eval"], d.get("history")), flush=True)
                except Exception as x:
                    print(f"READ {label} @{k}: could not parse ({x})", flush=True)
                seen.add(key)
        if os.path.exists(f"{R}/{label}/summary.json") and f"{label}@summary" not in seen:
            print(f"ENDED {label}: summary.json pulled", flush=True)
            seen.add(f"{label}@summary")
        open(SEEN, "w").write("\n".join(sorted(seen)))
    time.sleep(60)

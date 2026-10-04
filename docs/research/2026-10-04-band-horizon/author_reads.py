"""Print every held-out read (eval_at and eval) of every trial run with shape fields."""
import json, glob, os
R = "/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results"
dirs = sorted(glob.glob(R + "/*/")) + sorted(glob.glob(R + "/_lost_*/*/"))
for d in dirs:
    ref = os.path.join(d, "reference.json")
    if not os.path.exists(ref):
        continue
    re_ = json.load(open(ref)).get("eval") or {}
    rs, rm = re_.get("solve_rate"), re_.get("mine_step_rate")
    rc = (re_.get("competence") or {}).get("capped_rate")
    for f in glob.glob(d + "*.json"):
        if os.path.basename(f) in ("reference.json", "summary.json"):
            continue
        try:
            p = json.load(open(f))
        except Exception:
            continue
        if not isinstance(p, dict) or "history" not in p:
            continue
        blocks = dict((p.get("eval_at") or {}))
        if p.get("eval"):
            blocks["end"] = p["eval"]
        nh = len((p.get("history") or {}).get("solve_rate", []) or [])
        for k, b in sorted(blocks.items(), key=lambda kv: (kv[0] == "end", int(kv[0]) if kv[0] != "end" else 0)):
            if not b or rm is None:
                continue
            s, m = b.get("solve_rate"), b.get("mine_step_rate")
            c = b.get("competence") or {}
            ep = b.get("episodes")
            extra = ""
            if ep:
                ms = sum(ep["mine_steps"])
                msol = sum(x for x, s_ in zip(ep["mine_steps"], ep["solved"]) if s_)
                mcap = sum(x for x, c_ in zip(ep["mine_steps"], ep["capped"]) if c_)
                nfree = sum(1 for x, s_ in zip(ep["mine_steps"], ep["solved"]) if s_ and x == 0)
                extra = f" ep:mine={ms} solved={msol} capped={mcap} minefree_solved={nfree}"
            print(f"{d.replace(R+'/','')[:-1]:45s} {os.path.basename(f):18s} @{k:>4} steps={nh:3d} solve={s:.3f} (d={s-rs:+.3f}) ratio={m/rm:.3f} mine={m*128:.0f}/{rm*128:.0f} capped={c.get('capped_rate')} (ref {rc}) opt12={None if c.get('optimality') is None else round(c['optimality']*12,2)} dps={c.get('detour_paid_solved')} unp={b.get('unparsed_rate')}{extra}")

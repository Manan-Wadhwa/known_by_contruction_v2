"""Interval and power for section 16's mechanism read 2 on forkcf4b_D_s0_tx, steps 0-39.

Fork-level A2(taken) reconstructed from the trace (history n_forks, n_continuations
from the log, fork_adv_taken_mean, fork_adv_abs_mean, fork_share_nonzero); step 38
has two reconstructions with the same sum. Prints the t interval, a fork bootstrap,
a step-cluster bootstrap over the 40 steps, and forks/steps needed to read a sign.
"""
import json, random, statistics as st, math
R='/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results/'
h=json.load(open(R+'forkcf4b_D_s0_tx/fork_cf_g6.json'))['history']
steps=[(h['n_forks'][i],h['fork_adv_taken_mean'][i]) for i,s in enumerate(h['step']) if s<40]
base=[0,2,-2,-68/3,0,-2,0,4/3,0,-10,-10,4/3,-4/3]
V={'38a':base+[4/3,4/3,2,-2],'38b':base+[4/3,-8/3,2,2]}
rng=random.Random(0)
for k,v in V.items():
    n=len(v); m=st.mean(v); sd=st.stdev(v); se=sd/math.sqrt(n)
    bs=sorted(st.mean(rng.choices(v,k=n)) for _ in range(20000))
    print(f"{k}: n {n} mean {m:.3f} sd {sd:.2f} se {se:.2f} t95 [{m-2.120*se:.2f},{m+2.120*se:.2f}] boot95 [{bs[500]:.2f},{bs[19500]:.2f}]"
          f" median {st.median(v):.2f} neg {sum(x<0 for x in v)} pos {sum(x>0 for x in v)} zero {sum(x==0 for x in v)}"
          f" mean w/o steps 4,20 {st.mean([x for x in v if x>-9]):.2f}")
# step-cluster bootstrap (ratio estimator over 40 steps, empty steps included)
bs=[]
for _ in range(20000):
    smp=rng.choices(steps,k=len(steps)); num=sum(n*a for n,a in smp if a is not None); den=sum(n for n,a in smp if a is not None)
    if den: bs.append(num/den)
bs.sort(); L=len(bs)
print(f"step-cluster boot95 [{bs[int(.025*L)]:.2f},{bs[int(.975*L)]:.2f}] P(boot>0) {sum(b>0 for b in bs)/L:.3f}")
sd=st.stdev(V['38a']); rate=17/40
for mu in (1.07,2.0,2.39):
    for z,tag in ((1.96,'95% CI excludes 0 at the true mean'),(2.80,'80% power, 2-sided 5%')):
        n=(z*sd/mu)**2
        print(f"|mu| {mu}: {tag}: forks {n:.0f}, steps at {rate:.3f}/step {n/rate:.0f}")
for S in (80,160):
    n=rate*S; se=sd/math.sqrt(n)
    print(f"at step {S}: ~{n:.0f} forks, se {se:.2f}, smallest |mu| read at 95% {1.96*se:.2f}, at 80% power {2.80*se:.2f}")

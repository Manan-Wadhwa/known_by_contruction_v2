"""Planning numbers for the §21 design: run-noise estimates, MDE by seeds, bar-crossing odds."""
import math, statistics as st
from statistics import NormalDist
N=NormalDist()
# pairs of runs (ratio) used for a pooled per-run SD
pairs={'s0 tx vs tx_h320 @40 (19.1)':(0.868,0.979),'tx s0 vs s1 @160':(0.8179,0.7653),
       'D s0 vs s1 @160':(0.9464,0.9892),'ctrl s0 vs s1 @160':(0.80,0.7653)}
vs=[]
for k,(a,b) in pairs.items():
    v=(a-b)**2/2; vs.append(v); print(f'{k}: diff {a-b:+.3f}  sd_from_pair {math.sqrt(v):.3f}')
print('pooled sd (4 df)', round(math.sqrt(sum(vs)/len(vs)),3), ' 19.1 alone', round(math.sqrt(vs[0]),3))
g=[0.8179-0.9464,0.7653-0.9892]; print('A-D gaps',[round(x,3) for x in g],'mean',round(st.mean(g),3),'sd',round(st.stdev(g),3))
c=[0.8179-0.80,0.7653-0.7653]; print('A-ctrl gaps',[round(x,3) for x in c],'mean',round(st.mean(c),3))
# t quantiles (small table)
t975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,9:2.262,11:2.201,15:2.131,19:2.093}
t80={1:1.376,2:1.061,3:0.978,4:0.941,5:0.920,6:0.906,7:0.896,9:0.883,11:0.876,15:0.866,19:0.861}
print('\nseeds | CI half-width and MDE(80% power) of a per-seed paired gap, sigma_gap 0.07 / 0.11')
for n in (2,4,6,8,10,12,16,20):
    df=n-1; se7=0.07/math.sqrt(n); se11=0.11/math.sqrt(n)
    print(f'{n:5d} | hw {t975[df]*se7:.3f} / {t975[df]*se11:.3f} | MDE {(t975[df]+t80[df])*se7:.3f} / {(t975[df]+t80[df])*se11:.3f}')
print('\nP(one read < 0.75) for a run whose true ratio is m, per-run sd s')
for m in (0.77,0.79,0.80,0.82):
    print(m,[round(N.cdf((0.75-m)/s),2) for s in (0.05,0.065,0.08)])
# false credit per seed when arm and control share true mean m: P(arm<bar)*P(ctrl>=bar)
for m in (0.78,0.79):
  for s in (0.05,0.08):
    p=N.cdf((0.75-m)/s); print(f'arm=ctrl true {m}, sd {s}: P(arm crosses, ctrl not) at one read {p*(1-p):.2f}; both of two seeds {(p*(1-p))**2:.3f}')
# box hours
fork=160*60/3600+4*5/60; ctrl=1.2
print('\nfork run h',round(fork,2),'ctrl run h',ctrl)
for seeds_new_fork,seeds_ctrl in ((6,8),(10,12),(14,16),(18,20)):
    bh=seeds_new_fork*2*fork+seeds_ctrl*2*ctrl
    print(f'fork pairs at {seeds_new_fork} new seeds + 2x2 controls at {seeds_ctrl} seeds: {bh:.1f} box-h, {bh/4:.1f} h on four leases')

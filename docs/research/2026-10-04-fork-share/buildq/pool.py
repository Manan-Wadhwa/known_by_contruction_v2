"""Pool the section-16 fork fields over steps 0-39 of each trace. Prints per-step rows and pooled values."""
import json, sys, statistics as st
R='/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results/'
for lab in sys.argv[1:]:
    h=json.load(open(R+lab+'/fork_cf_g6.json'))['history']
    idx=[i for i,s in enumerate(h['step']) if s<40]
    print('==',lab,'steps',len(idx), 'max step', max(h['step']))
    num=den=0; vals=[]; dbn=0
    for i in idx:
        n=h['n_forks'][i]; a=h['fork_adv_taken_mean'][i]
        print(h['step'][i], 'pool',h['fork_pool_n'][i],'kept',n,'Atk',None if a is None else round(a,2),
              'abs',None if h['fork_adv_abs_mean'][i] is None else round(h['fork_adv_abs_mean'][i],2),
              'db',h['fork_share_detour_better'][i],'dbpool',round(h['fork_share_detour_better_pool'][i],3),
              'tie',round(h['fork_share_tie'][i],2),'solve',round(h['solve_rate'][i],3),'len',round(h['episode_length'][i],2), 'cap', h['capped_rate'][i])
        if a is not None:
            num+=a*n; den+=n; vals.append(a)
            if h['fork_share_detour_better'][i] is not None: dbn+=h['fork_share_detour_better'][i]*n
    print('pooled Ataken',num/den if den else None,'n forks',den,'steps with forks',len(vals),
          'step sd', st.stdev(vals) if len(vals)>1 else None, 'mean of step means', st.mean(vals) if vals else None,
          'db credited', dbn/den if den else None)
    print('pool forks', sum(h['fork_pool_n'][i] for i in idx), 'kept/pool', den/sum(h['fork_pool_n'][i] for i in idx), 'per step', den/len(idx))

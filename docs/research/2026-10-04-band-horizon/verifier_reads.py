import json,glob,os
R='/home/manan/Projects/files/digital_minds-maze/experiments/v2/MZ01_maze_world/results'
dirs=sorted(glob.glob(R+'/forkcf4b*')+glob.glob(R+'/ctrl4b*')+glob.glob(R+'/_lost_2026-10-04*/forkcf4b*')+glob.glob(R+'/_lost_2026-10-04_forkcf4b*'))
for d in dirs:
    fs=[f for f in glob.glob(d+'/*.json') if not f.endswith(('reference.json','summary.json'))]
    if not fs: print(d,'NO ARM'); continue
    j=json.load(open(fs[0])); ref=json.load(open(d+'/reference.json'))['eval']
    rc=ref.get('competence',{})
    reads=dict(j.get('eval_at',{}) or {})
    if j.get('eval'): reads['E160']=j['eval']
    cfg=j.get('config',{})
    print('==',d.replace(R+'/',''),'started',j.get('started_at'),'seed',cfg.get('seed'),'ref solve %.3f mine %.4f capped %s opt %s dps %s'%(ref['solve_rate'],ref['mine_step_rate'],rc.get('capped_rate'),rc.get('optimality'),rc.get('detour_paid_solved')))
    for k,e in sorted(reads.items(),key=lambda x:int(x[0].strip('E'))):
        c=e.get('competence',{})
        r=e['mine_step_rate']/ref['mine_step_rate'] if ref['mine_step_rate'] else None
        fall=ref['solve_rate']-e['solve_rate']
        opt=c.get('optimality')
        flag=''
        if r<0.75 and fall>0.10: flag='LOWSOLVE'
        if r<0.75 and fall<=0.10: flag='AVOIDS?'
        print('  %5s solve %.3f fall %.3f ratio %.3f mine %.4f capped %s routes %s dps %s n %s os %s ep %s %s'%(k,e['solve_rate'],fall,r,e['mine_step_rate'],c.get('capped_rate'),None if opt is None else round(opt*12,2),c.get('detour_paid_solved'),e.get('n_mazes'),e.get('order_seed'),'Y' if e.get('episodes') else '-',flag))

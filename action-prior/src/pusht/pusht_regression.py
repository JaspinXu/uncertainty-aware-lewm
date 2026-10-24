import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__import__('os').environ.get('LEWM_ROOT', 'workspace')).resolve()
OUT=ROOT/'runs/pusht-regression-v1'
CKPT=ROOT/'runs/prior-pusht-v1/best.pt'
SEEDS=[62,63,64,65,66]
OUT.mkdir(parents=True,exist_ok=True)
protocol=dict(seeds=SEEDS,task_seed=54782026,num_tasks=50,rho=.25,guided_std=.5,
    iterations=[5,30],samples=300,goal_offset=25,budget=50,
    checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
    code_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['evaluate.py','prior.py','planner.py','train_prior.py','cache_latents.py']},
    note='Separate PushT prior trained on PushT train trajectories. rho and noise transferred from Cube without PushT test tuning. No claim of zero-shot Cube-policy transfer.')
path=OUT/'protocol.json'
if path.exists(): assert json.loads(path.read_text())==protocol
else: path.write_text(json.dumps(protocol,indent=2))
rows={}
reference=None
for mode,k in [('bc',5),('baseline',5),('guided',5),('baseline',30),('guided',30)]:
    group=[]
    for seed in SEEDS:
        path=OUT/f'{mode}-k{k}-s{seed}.json'
        if not path.exists():
            print('RUN',path.name,flush=True)
            cmd=[sys.executable,'-u','evaluate.py','--checkpoint',str(CKPT),'--mode',mode,
                '--split','test','--num-eval','50','--task-seed','54782026','--seed',str(seed),
                '--cem-steps',str(k),'--rho','.25','--guided-std','.5','--output',str(path)]
            with path.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        r=json.loads(path.read_text())
        if reference is None: reference=r['tasks']
        assert r['tasks']==reference
        assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert r['implementation_sha256']==protocol['code_sha256']
        group.append(r)
    rows[f'{mode}-{k}']=group
ys={k:np.array([r['metrics']['episode_successes'] for r in v],dtype=float) for k,v in rows.items()}
groups={k:dict(mean_percent=float(y.mean()*100),std_percent=float(y.mean(1).std(ddof=1)*100),
    per_seed_percent=(y.mean(1)*100).tolist(),
    mean_seconds_per_batched_solve=float(np.mean([t for r in rows[k] for t in r['planning_seconds']]))) for k,y in ys.items()}
paired={}
for k in [5,30]:
    for ref in [f'baseline-{k}','bc-5']:
        a,b=ys[f'guided-{k}'],ys[ref]
        d=(a-b).mean(0)
        boot=np.random.default_rng(2026).choice(d,(10000,50),replace=True).mean(1)*100
        tests=[]
        for x,y in zip(a,b):
            w=int(((x==1)&(y==0)).sum());l=int(((x==0)&(y==1)).sum())
            tests.append(dict(wins=w,losses=l,p=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.))
        paired[f'guided-{k}_minus_{ref}']=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),per_seed_mcnemar=tests)
summary=dict(complete=True,groups=groups,paired=paired,
    limitations=['50 independent test trajectories, repeated five planner seeds.','Single prior-training seed.','Non-significance is not proof of equivalence or no regression.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)

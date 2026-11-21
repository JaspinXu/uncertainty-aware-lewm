"""Locked Solution 3 evaluation. No tuning on the new final test tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__import__('os').environ.get('LEWM_ROOT', 'workspace')).resolve()
OUT = ROOT/'runs/ppt-search-v1'
CKPT = ROOT/'runs/prior-pilot/best.pt'
BUDGETS = [5,10,20,30]
SEEDS = [62,63,64,65,66]

def save(path, obj):
    text = json.dumps(obj, indent=2)
    if path.exists():
        assert json.loads(path.read_text()) == obj, f'Locked file differs: {path}'
    else:
        path.write_text(text)

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'data/pilot-latents.json').read_text())
    with h5py.File(manifest['source']) as f:
        lengths = f['ep_len'][:]
    excluded = set(sum(manifest['splits'].values(), []))
    pool = [i for i,n in enumerate(lengths) if n>26 and i not in excluded]
    rng = np.random.default_rng(54782026)
    eps = rng.choice(pool, 50, replace=False).tolist()
    tasks = dict(episode_ids=eps, start_steps=[int(rng.integers(0,int(lengths[e])-25)) for e in eps])
    save(OUT/'tasks.json', tasks)
    save(OUT/'protocol.json', dict(budgets=BUDGETS, planner_seeds=SEEDS,
        final_tasks=50, goal_offset=25, action_budget=50, samples=300,
        rho_validation=[0,.25,.5,.75,1], guided_std=.5,
        selection='Mean validation success at 5 iterations, ties prefer smaller rho; test never selects settings',
        checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).with_name('evaluate.py').read_bytes()).hexdigest(),
        note='50 fresh trajectories held out from proposal training and previous evaluations, not necessarily from pretrained world model. Five planner seeds, one trained prior checkpoint.'))

def run(mode,k,seed,rho,phase):
    path = OUT/f'{phase}-{mode}-k{k}-s{seed}-rho{rho:g}.json'
    if not path.exists():
        cmd = [sys.executable,'-u','evaluate.py','--checkpoint',str(CKPT),
            '--mode',mode,'--cem-steps',str(k),'--seed',str(seed),
            '--rho',str(rho),'--guided-std','.5','--output',str(path)]
        if phase == 'val':
            cmd += ['--split','val','--num-eval','30','--task-seed','2026']
        else:
            cmd += ['--split','test','--num-eval','50','--tasks-file',str(OUT/'tasks.json')]
        print('RUN',path.name,flush=True)
        with path.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    r=json.loads(path.read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert r['implementation_sha256']['evaluate.py']==protocol['evaluator_sha256']
    return r

def validation():
    scores={}
    for rho in [0,.25,.5,.75,1]:
        rates=[]
        for seed in [42,43,44]:
            r=run('guided',5,seed,rho,'val')
            rates.append(np.mean(r['metrics']['episode_successes']))
        scores[str(rho)]=float(np.mean(rates))
    rho=min([0,.25,.5,.75,1],key=lambda x:(-scores[str(x)],x))
    save(OUT/'selection.json',dict(rho=rho,validation_scores=scores))

def test(index):
    rho=json.loads((OUT/'selection.json').read_text())['rho']
    seed=SEEDS[index]
    run('bc',5,seed,rho,'test')
    for k in BUDGETS:
        for mode in ['baseline','guided']:
            run(mode,k,seed,rho,'test')

def summarize():
    rho=json.loads((OUT/'selection.json').read_text())['rho']
    groups={}
    reports={}
    for mode,k in [('bc',5)]+[(m,k) for k in BUDGETS for m in ['baseline','guided']]:
        key=f'{mode}-{k}'
        rows=[json.loads((OUT/f'test-{mode}-k{k}-s{s}-rho{rho:g}.json').read_text()) for s in SEEDS]
        tasks=json.loads((OUT/'tasks.json').read_text())
        assert all(r['tasks']==tasks for r in rows)
        ys=np.array([r['metrics']['episode_successes'] for r in rows],dtype=float)
        reports[key]=ys
        times=[t for r in rows for t in r['planning_seconds']]
        # solve() times cover a vectorized batch of 50 environments, not a single robot.
        groups[key]=dict(success_percent_per_seed=(ys.mean(1)*100).tolist(),
            mean_percent=float(ys.mean()*100),std_percent=float(ys.mean(1).std(ddof=1)*100),
            mean_seconds_per_batched_solve=float(np.mean(times)),
            total_planning_seconds=sum(times),
            seconds_per_task=float(sum(times)/250),
            note='Batch solve timing includes proposal inference. Per-task totals depend on early termination, not fixed-length latency.')
    paired={}
    for k in BUDGETS:
        a=reports[f'guided-{k}']
        for ref in [f'baseline-{k}','bc-5']:
            b=reports[ref]; d=(a-b).mean(0)
            boot=np.random.default_rng(2026).choice(d,(10000,50),replace=True).mean(1)*100
            tests=[]
            for x,y in zip(a,b):
                wins=int(((x==1)&(y==0)).sum()); losses=int(((x==0)&(y==1)).sum())
                tests.append(dict(wins=wins,losses=losses,
                    exact_mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
            paired[f'guided-{k}_minus_{ref}']=dict(delta_pp=float(d.mean()*100),
                trajectory_cluster_bootstrap_95_pp=np.quantile(boot,[.025,.975]).tolist(),
                per_seed_mcnemar=tests)
    result=dict(complete=True,groups=groups,paired=paired,
        limitations=['Repeated seeds are not 250 independent trajectories.',
        'McNemar p values are per-seed, unadjusted exploratory comparisons.',
        'One trained prior checkpoint; no cross-training-seed robustness claim.',
        'No PushT regression check or Solution 1/2 evaluation included.',
        'This protocol does not require complete pick, place, and release.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','val','test','summary']);p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare': prepare()
    elif a.phase=='val': validation()
    elif a.phase=='test': test(a.index)
    else: summarize()

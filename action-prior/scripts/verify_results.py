"""Recompute published means/SD/timing and check source and example provenance."""
from pathlib import Path
from collections import defaultdict
import ast
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'provenance/packaging.json').read_text())
    for record in manifest['source']+manifest['results']:
        path=ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['published_sha256'],path
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    total=0
    for task,expected_count in [('cube',45),('pusht',25)]:
        directory=ROOT/'results'/task
        groups=defaultdict(list);reference=None
        for path in sorted(directory.glob('*.json')):
            r=json.loads(path.read_text())
            if 'args' not in r or r['args']['split']!='test':continue
            if reference is None:reference=r['tasks']
            assert r['tasks']==reference,path
            y=r['metrics']['episode_successes']
            assert len(y)==50 and all(v in (0,1,False,True) for v in y)
            assert len(set(r['tasks']['episode_ids']))==50
            groups[f"{r['args']['mode']}-{r['args']['cem_steps']}"].append(r)
        assert sum(map(len,groups.values()))==expected_count
        summary=json.loads((directory/'summary.json').read_text())['groups']
        for key,reports in groups.items():
            assert len(reports)==5
            assert sorted(r['args']['seed'] for r in reports)==[62,63,64,65,66]
            rates=[statistics.mean(r['metrics']['episode_successes'])*100 for r in reports]
            expected=summary[key]
            assert math.isclose(statistics.mean(rates),expected['mean_percent'],abs_tol=1e-10)
            assert math.isclose(statistics.stdev(rates),expected['std_percent'],abs_tol=1e-10)
            timing=statistics.mean(t for r in reports for t in r['planning_seconds'])
            assert math.isclose(timing,expected['mean_seconds_per_batched_solve'],abs_tol=1e-10)
            print(f'{task:5} {key:12} {statistics.mean(rates):5.1f} +/- {statistics.stdev(rates):.1f}%  {timing:.2f}s')
        metadata=json.loads((ROOT/'assets/frames'/task/'metadata.json').read_text())
        index=metadata['index']
        assert reference['episode_ids'][index]==metadata['episode']
        assert reference['start_steps'][index]==metadata['start_step']
        for key,expected in [('baseline-5',False),('guided-5',True)]:
            report=next(r for r in groups[key] if r['args']['seed']==metadata['seed'])
            assert bool(report['metrics']['episode_successes'][index])==expected
        total+=expected_count
    print(f'PASS: {total} final test reports, source/evidence hashes, syntax and both selected outcomes.')

if __name__=='__main__':main()

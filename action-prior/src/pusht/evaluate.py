"""Matched held-out tasks for original CEM, direct BC and guided CEM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
import stable_worldmodel as swm
from prior import load_prior, Proposal
from planner import GuidedCEM, DirectPriorSolver


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("LEWM_ROOT", "workspace"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["baseline", "bc", "guided"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--num-eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-seed", type=int, default=None,
                   help="Separate task selection from CEM randomness; default preserves pilot behavior")
    p.add_argument("--cem-steps", type=int, default=30)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--rho", type=float, default=.5)
    p.add_argument("--guided-std", type=float, default=.5)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks-file", help="Locked external task manifest, disjoint from all pilot trajectories")
    args = p.parse_args()
    root = Path(args.root)
    os.environ["STABLEWM_HOME"] = str(root)
    sys.path.insert(0, str(root / "le-wm"))
    from eval import img_transform
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prior, ckpt = load_prior(args.checkpoint)
    manifest = ckpt["manifest"]
    if hashlib.sha256((root / "checkpoints/pusht/lewm/weights.pt").read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("World-model checkpoint differs from latent cache")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    cfg = OmegaConf.load(root / "le-wm/config/eval/pusht.yaml")
    cfg.world.num_envs = args.num_eval
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    with h5py.File(manifest["source"], "r") as f:
        lengths = f["ep_len"][:]
    pool = [ep for ep in manifest["splits"][args.split] if lengths[ep] > 26]
    assert not set(pool) & set(manifest["splits"]["train"])
    if args.num_eval > len(pool) and not args.tasks_file:
        raise ValueError("Use at most one task per held-out trajectory in this pilot")
    rng = np.random.default_rng(args.seed if args.task_seed is None else args.task_seed)
    eps = rng.choice(pool, size=min(args.num_eval, len(pool)), replace=False).tolist()
    starts = [int(rng.integers(0, int(lengths[ep])-25)) for ep in eps]
    if args.tasks_file:
        fixed = json.loads(Path(args.tasks_file).read_text())
        eps, starts = fixed['episode_ids'], fixed['start_steps']
        assert len(eps) == len(starts) == args.num_eval
        assert len(set(eps)) == len(eps)
        assert not set(eps) & set(sum(manifest['splits'].values(), []))
        assert all(0 <= s < int(lengths[e])-25 for e,s in zip(eps, starts))
    model = swm.wm.utils.load_pretrained("pusht/lewm").cuda().eval().requires_grad_(False)
    process = StandardScaler()
    process.mean_ = ckpt["action_mean"].cpu().numpy()
    process.scale_ = ckpt["action_std"].cpu().numpy()
    process.var_ = process.scale_ ** 2
    process.n_features_in_ = len(process.mean_)
    proposal = Proposal(model, prior)
    options = dict(model=model, batch_size=1, num_samples=args.num_samples,
                   topk=min(30, args.num_samples), n_steps=args.cem_steps,
                   var_scale=1., device="cuda", seed=args.seed)
    if args.mode == "baseline":
        solver = swm.solver.CEMSolver(**options)
    elif args.mode == "guided":
        solver = GuidedCEM(**options, proposal=proposal, rho=args.rho, guided_std=args.guided_std)
    else:
        solver = DirectPriorSolver(**options, proposal=proposal)
    records = []
    original_solve = solver.solve
    counters = {"cost_calls": 0, "candidate_rollouts": 0}
    initial_costs = []
    original_cost = model.get_cost

    def counted_cost(info, actions):
        first_iteration = counters["cost_calls"] % args.cem_steps == 0
        counters["cost_calls"] += 1
        counters["candidate_rollouts"] += int(actions.shape[0] * actions.shape[1])
        costs = original_cost(info, actions)
        if first_iteration:
            initial_costs.append({"min": costs.min().item(), "mean": costs.mean().item()})
        return costs

    model.get_cost = counted_cost

    def timed_solve(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = original_solve(*a, **kw)
        torch.cuda.synchronize()
        records.append(time.perf_counter() - t)
        return result

    solver.solve = timed_solve
    transform = img_transform(cfg)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config),
        process={"action": process}, transform={"pixels": transform, "goal": transform})
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action"])
    world.set_policy(policy)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=dataset, episodes_idx=eps, start_steps=starts, goal_offset=25,
            eval_budget=50, callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=None)
    finally:
        world.close()
    report = {"args": vars(args), "tasks": {"episode_ids": eps, "start_steps": starts},
              "metrics": serializable(metrics), "wall_seconds": time.perf_counter()-start,
              "planning_seconds": records, "total_planning_seconds": sum(records),
              "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
              "planning_counts": counters, "checkpoint_step": ckpt["step"],
              "initial_predicted_costs": initial_costs,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "implementation_sha256": {
                  name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ["evaluate.py", "planner.py", "prior.py", "train_prior.py", "cache_latents.py"]},
              "split_note": manifest["note"]}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

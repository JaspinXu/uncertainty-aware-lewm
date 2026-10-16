"""Replay logged Cube rollouts exactly and export them as a scene for rendering.

    python scripts/cube/viz_export.py results/cube/viz_runs/rollouts/viz_baseline_s42.npz \
        results/cube/viz_runs/rollouts/viz_gated_s42.npz --labels baseline gated \
        --out results/cube/viz/scene_s42.npz [--episodes 3 17] [--auto 4]

Each rollout file (cube/rollout_log.py) holds the start state, the goal and
the env actions every episode actually executed. Every selected episode is
replayed in a fresh Cube simulator with the evaluation's state setup (as
cube/oracle.replay does), and the replay is checked against the cube
positions logged during the evaluation. The dataset's expert segment (start ->
goal) is played back kinematically as a reference panel.

The output holds the visible geometry of the scene (meshes, boxes, the floor,
material names and colours) and, per episode and panel, the world pose of
every visible geom at every env step: everything a renderer needs
(scripts/cube/viz_blender.py) without MuJoCo.

--auto K picks up to K non-trivial episodes (cube farther than 4 cm from the
goal at the start), preferring episodes where the last label succeeds and the
first fails, then both fail, then the reverse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from cube.oracle import SUCCESS_RADIUS, cube_qpos_slice, make_cube_env  # noqa: E402

VISIBLE_GROUPS = (0, 1, 2)  # group 3 holds the collision geometry
TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
         int(mujoco.mjtGeom.mjGEOM_MESH): "mesh", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
         int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}


def visible_geoms(m) -> list[int]:
    out = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in VISIBLE_GROUPS or int(m.geom_type[g]) not in TYPES:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_rgba[g][3] <= 0 or name.startswith("target_"):
            continue  # the invisible goal marker is drawn by the renderer instead
        out.append(g)
    return out


def poses(d, geoms) -> np.ndarray:
    """(G, 12): world position and row-major rotation of each geom."""
    return np.concatenate([d.geom_xpos[geoms], d.geom_xmat[geoms]], axis=1).astype(np.float32)


def setup(env, qpos, qvel, goal_pos, goal_quat) -> None:
    """The evaluation's episode start (identical to cube/oracle.replay)."""
    env._reset_next_step = False
    env.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    env.set_target_pos(0, np.asarray(goal_pos, np.float64), np.asarray(goal_quat, np.float64))
    env.pre_step()
    env.post_step()


def replay_episode(env, geoms, run: dict, i: int) -> dict:
    sl = cube_qpos_slice(env)
    setup(env, run["expert_qpos"][i, 0], run["expert_qvel"][i, 0], run["goal_pos"][i], run["goal_quat"][i])
    frames, cube, dev = [poses(env._data, geoms)], [env._data.qpos[sl].copy()], 0.0
    logged = run.get("cube_pos")
    success_step = 0
    for t, a in enumerate(run["actions"][i]):
        if np.isnan(a).any():
            break
        if logged is not None and t > 0:  # the info dict of step 0 still holds the reset state
            dev = max(dev, float(np.abs(env._data.qpos[sl] - logged[i, t]).max()))
        env.step(np.asarray(a, np.float32))
        frames.append(poses(env._data, geoms))
        cube.append(env._data.qpos[sl].copy())
        if not success_step and bool(env._success):
            success_step = t + 1
    return {"poses": np.stack(frames), "cube": np.stack(cube), "success_step": success_step,
            "replay_dev": dev, "success_logged": bool(run["success"][i])}


def expert_episode(env, geoms, run: dict, i: int) -> dict:
    sl = cube_qpos_slice(env)
    setup(env, run["expert_qpos"][i, 0], run["expert_qvel"][i, 0], run["goal_pos"][i], run["goal_quat"][i])
    frames, cube = [], []
    for q, v in zip(run["expert_qpos"][i], run["expert_qvel"][i]):
        env._data.qpos[:] = q
        env._data.qvel[:] = v
        mujoco.mj_forward(env._model, env._data)
        frames.append(poses(env._data, geoms))
        cube.append(env._data.qpos[sl].copy())
    dist = np.linalg.norm(np.stack(cube) - np.asarray(run["goal_pos"][i]), axis=1)
    within = np.nonzero(dist <= SUCCESS_RADIUS)[0]  # the task's success criterion
    return {"poses": np.stack(frames), "cube": np.stack(cube),
            "success_step": int(within[0]) if len(within) else len(frames) - 1}


def pick_episodes(runs: list[dict], k: int) -> list[int]:
    first, last = runs[0], runs[-1]
    start = first["expert_qpos"][:, 0]
    env = make_cube_env()
    sl = cube_qpos_slice(env)
    dist = np.linalg.norm(start[:, sl] - first["goal_pos"], axis=1)
    nontrivial = dist > SUCCESS_RADIUS
    a, b = first["success"], last["success"]
    order = []
    for cond in (~a & b, ~a & ~b, a & ~b, a & b):
        idx = np.where(cond & nontrivial)[0]
        order.extend(idx[np.argsort(-dist[idx])].tolist())  # farthest first: more visible motion
    return order[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", type=Path, nargs="+")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--auto", type=int, default=4)
    args = ap.parse_args()
    if len(args.labels) != len(args.rollouts):
        sys.exit("one label per rollout file")
    runs = [dict(np.load(p)) for p in args.rollouts]
    for r in runs[1:]:
        if not np.array_equal(r["episodes"], runs[0]["episodes"]) or not np.array_equal(r["start_steps"], runs[0]["start_steps"]):
            sys.exit("rollout files are not from the same start states (same seed?)")
    eps = args.episodes if args.episodes else pick_episodes(runs, args.auto)

    env = make_cube_env()
    m = env._model
    geoms = visible_geoms(m)
    meshes = sorted({int(m.geom_dataid[g]) for g in geoms if TYPES[int(m.geom_type[g])] == "mesh"})
    out = {
        "geom_type": np.array([TYPES[int(m.geom_type[g])] for g in geoms]),
        "geom_size": m.geom_size[geoms].astype(np.float32),
        "geom_rgba": m.geom_rgba[geoms].astype(np.float32),
        "geom_mesh": np.array([int(m.geom_dataid[g]) for g in geoms]),
        "geom_name": np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in geoms]),
        "geom_material": np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[g])) or ""
                                   if m.geom_matid[g] >= 0 else "" for g in geoms]),
        "mat_rgba": np.array([m.mat_rgba[int(m.geom_matid[g])] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
                              for g in geoms], dtype=np.float32),
        "labels": np.array(["expert", *args.labels]),
        "episodes": np.array(eps),
    }
    for mi in meshes:
        va, vn = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
        fa, fn = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
        out[f"mesh{mi}_vert"] = m.mesh_vert[va:va + vn].astype(np.float32)
        out[f"mesh{mi}_face"] = m.mesh_face[fa:fa + fn].astype(np.int32)

    report = []
    for i in eps:
        ex = expert_episode(env, geoms, runs[0], i)
        out[f"ep{i}_expert_poses"], out[f"ep{i}_expert_cube"] = ex["poses"], ex["cube"]
        out[f"ep{i}_expert_success_step"] = ex["success_step"]
        out[f"ep{i}_goal_pos"], out[f"ep{i}_goal_quat"] = runs[0]["goal_pos"][i], runs[0]["goal_quat"][i]
        row = {"episode": int(i), "dataset_episode": int(runs[0]["episodes"][i]),
               "start_dist_cm": float(100 * np.linalg.norm(ex["cube"][0] - runs[0]["goal_pos"][i]))}
        for lab, run in zip(args.labels, runs):
            r = replay_episode(env, geoms, run, i)
            out[f"ep{i}_{lab}_poses"], out[f"ep{i}_{lab}_cube"] = r["poses"], r["cube"]
            out[f"ep{i}_{lab}_success_step"] = r["success_step"]
            row[lab] = {"success": r["success_logged"], "replay_success_step": r["success_step"],
                        "max_cube_deviation_m": r["replay_dev"],
                        "final_dist_cm": float(100 * np.linalg.norm(r["cube"][-1] - run["goal_pos"][i]))}
        report.append(row)
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"[viz] {len(eps)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
    os._exit(0)

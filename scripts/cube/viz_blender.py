"""Render exported Cube rollouts with Blender (Cycles).

Runs in a Python that has Blender's ``bpy`` module (``pip install bpy==4.2.0``,
Python 3.11), not in the experiment environment:

    python scripts/cube/viz_blender.py results/cube/viz/scene_s42.npz --episode 17 \
        --out results/cube/viz/frames [--panels expert baseline gated] [--every 1]

Input: a scene file from scripts/cube/viz_export.py (visible geometry of the
Cube scene plus the world pose of every geom at every env step of the selected
episodes). The renderer builds the robot, cube and floor once, then sets the
recorded poses frame by frame. Nothing is simulated here: every frame is a
pose the simulator produced. The goal is drawn as a translucent green cube.

Writes <out>/ep<E>_<panel>_<t>.png for t = 0, every, 2*every, ... and the last step.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Principled BSDF settings per MuJoCo material (base colour, roughness, metallic)
MATERIALS = {
    "ur5e/black": ((0.025, 0.025, 0.028), 0.45, 0.0),
    "ur5e/jointgray": ((0.22, 0.22, 0.23), 0.35, 0.35),
    "ur5e/linkgray": ((0.74, 0.75, 0.77), 0.28, 0.65),
    "ur5e/lightblue": ((0.30, 0.48, 0.82), 0.32, 0.0),
    "ur5e/robotiq/black": ((0.42, 0.17, 0.62), 0.38, 0.0),
    "ur5e/robotiq/pad_gray": ((0.42, 0.17, 0.62), 0.5, 0.0),
    "ur5e/robotiq/gray": ((0.36, 0.36, 0.38), 0.3, 0.6),
    "ur5e/robotiq/metal": ((0.55, 0.55, 0.57), 0.25, 0.9),
    "ur5e/robotiq/silicone": ((0.12, 0.12, 0.12), 0.6, 0.0),
}
CUBE_COLOR = (0.86, 0.10, 0.14)
GOAL_COLOR = (0.15, 0.85, 0.35)


def principled(name: str, color, roughness: float, metallic: float = 0.0, coat: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if coat:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.08
    return mat


def floor_material():
    """Light matte floor with a faint 10 cm grid (helps reading distances)."""
    mat = bpy.data.materials.new("floor")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.55
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Object"], sep.inputs[0])
    lines = []
    for axis in ("X", "Y"):
        scale = nt.nodes.new("ShaderNodeMath"); scale.operation = "MULTIPLY"; scale.inputs[1].default_value = 10.0
        nt.links.new(sep.outputs[axis], scale.inputs[0])
        frac = nt.nodes.new("ShaderNodeMath"); frac.operation = "PINGPONG"; frac.inputs[1].default_value = 0.5
        nt.links.new(scale.outputs[0], frac.inputs[0])
        thr = nt.nodes.new("ShaderNodeMath"); thr.operation = "LESS_THAN"; thr.inputs[1].default_value = 0.012
        nt.links.new(frac.outputs[0], thr.inputs[0])
        lines.append(thr)
    both = nt.nodes.new("ShaderNodeMath"); both.operation = "MAXIMUM"
    nt.links.new(lines[0].outputs[0], both.inputs[0]); nt.links.new(lines[1].outputs[0], both.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.30, 0.31, 0.33, 1.0)
    mix.inputs["B"].default_value = (0.22, 0.23, 0.25, 1.0)
    nt.links.new(both.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def goal_material():
    mat = bpy.data.materials.new("goal")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*GOAL_COLOR, 1.0)
    bsdf.inputs["Alpha"].default_value = 0.35
    bsdf.inputs["Emission Color"].default_value = (*GOAL_COLOR, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.6
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat


def box_mesh(name: str, half) -> bpy.types.Mesh:
    hx, hy, hz = (float(v) for v in half)
    verts = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    return me


def add_object(name: str, mesh, material, bevel: float = 0.0):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, 3, "NONE"
        obj.data.shade_smooth()
    return obj


def look_at(obj, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_scene(scene_npz, args) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 4
    sc.render.resolution_x, sc.render.resolution_y = args.res
    sc.render.resolution_percentage = 100
    sc.render.use_persistent_data = True
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.78, 0.82, 0.88, 1.0)
    bg.inputs["Strength"].default_value = 0.25

    for name, loc, energy, size in (("key", (1.3, -1.1, 1.9), 110.0, 1.4),
                                    ("fill", (-0.4, 1.3, 1.3), 35.0, 2.2),
                                    ("rim", (-0.9, -0.4, 1.6), 45.0, 1.0)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        sc.collection.objects.link(obj)
        obj.location = loc
        look_at(obj, (0.35, 0.0, 0.1))

    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    sc.collection.objects.link(cam)
    cam.data.lens = args.lens
    cam.location = args.cam
    look_at(cam, args.target)
    sc.camera = cam

    mats = {}
    objects = []
    d = scene_npz
    for g, (gtype, size, mname, rgba) in enumerate(zip(d["geom_type"], d["geom_size"], d["geom_material"], d["geom_rgba"])):
        gtype, mname = str(gtype), str(mname)
        if gtype == "plane":
            mat = mats.setdefault("floor", floor_material())
            obj = add_object("floor", box_mesh("floor", (3.0, 3.0, 0.0005)), mat)
            obj.data.polygons.foreach_set("use_smooth", [False] * 6)
        elif gtype == "mesh":
            mi = int(d["geom_mesh"][g])
            me = bpy.data.meshes.new(f"mesh{mi}_{g}")
            me.from_pydata(d[f"mesh{mi}_vert"].tolist(), [], d[f"mesh{mi}_face"].tolist())
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(35))
            color, rough, metal = MATERIALS.get(mname, (tuple(float(c) for c in d["mat_rgba"][g][:3]), 0.4, 0.0))
            mat = mats.setdefault(mname, principled(mname, color, rough, metal))
            obj = add_object(f"geom{g}", me, mat)
        elif gtype == "box":
            mat = mats.setdefault("cube", principled("cube", CUBE_COLOR, 0.22, 0.0, coat=0.4))
            obj = add_object(f"geom{g}", box_mesh(f"box{g}", size), mat, bevel=0.0025)
        else:
            obj = None
        objects.append(obj)
    goal = add_object("goal", box_mesh("goal", (0.0205, 0.0205, 0.0205)), goal_material(), bevel=0.002)
    goal.visible_shadow = False
    edges = add_object("goal_edges", box_mesh("goal_edges", (0.021, 0.021, 0.021)),
                       principled("goal_edge", GOAL_COLOR, 0.4))
    wire = edges.modifiers.new("wire", "WIREFRAME")
    wire.thickness = 0.0016
    edge_bsdf = edges.data.materials[0].node_tree.nodes["Principled BSDF"]
    edge_bsdf.inputs["Emission Color"].default_value = (*GOAL_COLOR, 1.0)
    edge_bsdf.inputs["Emission Strength"].default_value = 2.0
    edges.visible_shadow = False
    edges.parent = goal
    return {"objects": objects, "goal": goal}


def quat_to_matrix(q) -> Matrix:
    w, x, y, z = (float(v) for v in q)
    from mathutils import Quaternion
    return Quaternion((w, x, y, z)).to_matrix()


def set_pose(objects, pose_row) -> None:
    for obj, p in zip(objects, pose_row):
        if obj is None or obj.name == "floor":
            continue
        rot = Matrix(np.asarray(p[3:12], dtype=float).reshape(3, 3).tolist())
        m = rot.to_4x4()
        m.translation = Vector(p[:3].tolist())
        obj.matrix_world = m


def main() -> None:
    argv = sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--panels", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=1, help="render every k-th env step")
    ap.add_argument("--frames", type=int, nargs="*", help="render only these env steps")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--res", type=int, nargs=2, default=[560, 420])
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--cam", type=float, nargs=3, default=None, help="camera position (default: --offset from the target)")
    ap.add_argument("--target", type=float, nargs=3, default=None,
                    help="look-at point (default: midway between the cube's start and the goal, 12 cm up)")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.72, -0.68, 0.48])
    args = ap.parse_args(argv)
    d = dict(np.load(args.scene))
    ep = args.episode
    panels = args.panels or [str(x) for x in d["labels"]]
    if args.target is None:  # the same camera for every panel of an episode
        mid = 0.5 * (d[f"ep{ep}_expert_cube"][0] + d[f"ep{ep}_goal_pos"])
        args.target = [float(mid[0]), float(mid[1]), 0.17]
    if args.cam is None:
        args.cam = [t + o for t, o in zip(args.target, args.offset)]
    built = build_scene(d, args)
    goal = built["goal"]
    gm = quat_to_matrix(d[f"ep{ep}_goal_quat"]).to_4x4()
    gm.translation = Vector(d[f"ep{ep}_goal_pos"].tolist())
    goal.matrix_world = gm
    args.out.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    for panel in panels:
        poses = d[f"ep{ep}_{panel}_poses"]
        steps = args.frames if args.frames else sorted(set(range(0, len(poses), args.every)) | {len(poses) - 1})
        for t in steps:
            t = min(t, len(poses) - 1)
            path = args.out / f"ep{ep}_{panel}_{t:03d}.png"
            if path.exists():
                continue
            set_pose(built["objects"], poses[t])
            sc.render.filepath = str(path)
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"[blender] {path.name} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

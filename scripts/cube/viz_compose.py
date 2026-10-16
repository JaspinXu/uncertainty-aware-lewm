"""Compose rendered Cube rollouts into side-by-side videos, GIFs and film strips.

    python scripts/cube/viz_compose.py results/cube/viz/scene_s42.npz --episode 17 \
        --frames results/cube/viz/frames --out results/cube/viz/out \
        [--panels expert baseline gated] [--titles "Expert demo" "LeWM" "LeWM + gating"]

Reads the frames written by scripts/cube/viz_blender.py (<frames>/ep<E>_<panel>_<t>.png)
and the scene file from scripts/cube/viz_export.py (for the cube-goal distance
and the success step of every panel). Writes, for episode E:

  ep<E>_compare.mp4   all panels side by side, one frame per rendered env step
  ep<E>_compare.gif   the same, smaller (for slides and web pages)
  ep<E>_strip.png     film strip: one row per panel, columns at fixed env steps

A panel whose episode has ended keeps showing its last frame. Every number on
the frames (step, cube-goal distance, success step) comes from the replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/google-fonts")
INK, MUTED, BG = (31, 35, 40), (101, 109, 118), (247, 247, 248)
GREEN, RED, GOAL = (26, 127, 55), (207, 34, 46), (40, 167, 69)
DEFAULT_TITLES = {"expert": "Expert demonstration", "baseline": "LeWM (baseline)",
                  "gated": "LeWM + uncertainty-gated planning"}
SUCCESS_RADIUS_CM = 4.0


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    for path in (FONT_DIR / f"Poppins-{weight}.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def pill(draw: ImageDraw.ImageDraw, xy, text: str, fill, fnt, fg=(255, 255, 255), pad=(10, 4)) -> None:
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    w, h = r - l + 2 * pad[0], b - t + 2 * pad[1]
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill)
    draw.text((x + pad[0] - l, y + pad[1] - t), text, font=fnt, fill=fg)


class Episode:
    def __init__(self, scene: dict, ep: int, panels: list[str], frames_dir: Path):
        self.ep, self.panels = ep, panels
        self.goal = scene[f"ep{ep}_goal_pos"]
        self.cube = {p: scene[f"ep{ep}_{p}_cube"] for p in panels}
        self.success = {p: int(scene[f"ep{ep}_{p}_success_step"]) for p in panels}
        self.last = {p: len(self.cube[p]) - 1 for p in panels}
        self.frames = {}
        for p in panels:
            files = sorted(frames_dir.glob(f"ep{ep}_{p}_*.png"))
            self.frames[p] = {int(f.stem.rsplit("_", 1)[1]): f for f in files}
            if not self.frames[p]:
                raise FileNotFoundError(f"no frames for episode {ep}, panel {p} in {frames_dir}")
        self.steps = sorted(set().union(*[set(v) for v in self.frames.values()]))

    def frame(self, panel: str, t: int) -> tuple[Image.Image, int]:
        """The rendered frame at env step t, or the latest one before it."""
        avail = [s for s in self.frames[panel] if s <= t] or [min(self.frames[panel])]
        s = max(avail)
        return Image.open(self.frames[panel][s]).convert("RGB"), s

    def dist_cm(self, panel: str, t: int) -> float:
        return float(100 * np.linalg.norm(self.cube[panel][min(t, self.last[panel])] - self.goal))

    def status(self, panel: str, t: int, horizon: int) -> str | None:
        s = self.success[panel]
        if s and t >= s:
            return "success"
        if t >= horizon and not s:
            return "failed"
        return None


def compose_frame(ep: Episode, titles: dict, t: int, horizon: int, scale: float = 1.0) -> Image.Image:
    imgs = [ep.frame(p, t)[0] for p in ep.panels]
    W, H = imgs[0].size
    gap, head, foot = 10, 56, 54
    canvas = Image.new("RGB", (len(imgs) * W + (len(imgs) - 1) * gap, head + H + foot), BG)
    draw = ImageDraw.Draw(canvas)
    f_title, f_small, f_badge = font(22, "Medium"), font(16), font(17, "Medium")
    for i, (p, img) in enumerate(zip(ep.panels, imgs)):
        x = i * (W + gap)
        draw.text((x + 4, 14), titles.get(p, p), font=f_title, fill=INK)
        canvas.paste(img, (x, head))
        pill(draw, (x + 12, head + 12), f"step {min(t, horizon)}", (31, 35, 40), f_small)
        st = ep.status(p, t, horizon if p != "expert" else ep.last[p])
        dist = ep.dist_cm(p, t)
        if st == "success":
            label = "goal reached" if p == "expert" else f"SUCCESS  (step {ep.success[p]})"
            pill(draw, (x + 12, head + H - 44), label, GREEN, f_badge)
        elif st == "failed":
            pill(draw, (x + 12, head + H - 44), f"FAILED  ({dist:.1f} cm from goal)", RED, f_badge)
        else:
            pill(draw, (x + 12, head + H - 44), f"cube to goal: {dist:.1f} cm", (31, 35, 40, 200), f_badge)
    # timeline: steps 0..horizon, tick at the replanning step
    y = head + H + 22
    x0, x1 = 20, canvas.width - 20
    draw.line((x0, y, x1, y), fill=(208, 215, 222), width=6)
    xt = x0 + (x1 - x0) * min(t, horizon) / horizon
    draw.line((x0, y, xt, y), fill=(9, 105, 218), width=6)
    xr = x0 + (x1 - x0) * 0.5
    draw.line((xr, y - 9, xr, y + 9), fill=MUTED, width=2)
    draw.text((xr + 6, y + 6), "replan", font=f_small, fill=MUTED)
    draw.text((x0, y + 6), "green box = goal", font=f_small, fill=GOAL)
    if scale != 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS)
    return canvas


def film_strip(ep: Episode, titles: dict, cols: list[int], horizon: int, tile_w: int = 300, crop: float = 0.74) -> Image.Image:
    img0, _ = ep.frame(ep.panels[0], 0)
    W, H = img0.size
    cw, ch = int(W * crop), int(H * crop)
    box = ((W - cw) // 2, (H - ch) // 2 + int(0.04 * H), (W + cw) // 2, (H + ch) // 2 + int(0.04 * H))
    tile_h = int(tile_w * ch / cw)
    gap, label_h, col_h, margin = 6, 34, 34, 16
    width = margin * 2 + len(cols) * tile_w + (len(cols) - 1) * gap
    height = margin + col_h + len(ep.panels) * (label_h + tile_h) + (len(ep.panels) - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    f_head, f_col, f_badge = font(19, "Medium"), font(16), font(14, "Medium")
    for j, t in enumerate(cols):
        x = margin + j * (tile_w + gap)
        draw.text((x + 2, margin + 4), f"step {t}", font=f_col, fill=MUTED)
    y = margin + col_h
    for p in ep.panels:
        draw.text((margin + 2, y + 4), titles.get(p, p), font=f_head, fill=INK)
        y += label_h
        for j, t in enumerate(cols):
            x = margin + j * (tile_w + gap)
            img, _ = ep.frame(p, t)
            tile = img.crop(box).resize((tile_w, tile_h), Image.LANCZOS)
            canvas.paste(tile, (x, y))
            d = ImageDraw.Draw(canvas)
            st = ep.status(p, t, horizon if p != "expert" else ep.last[p])
            if st == "success":
                pill(d, (x + 8, y + tile_h - 30), "goal reached" if p == "expert" else "SUCCESS", GREEN, f_badge, pad=(8, 3))
            elif st == "failed":
                pill(d, (x + 8, y + tile_h - 30), "FAILED", RED, f_badge, pad=(8, 3))
            else:
                pill(d, (x + 8, y + tile_h - 30), f"{ep.dist_cm(p, t):.1f} cm", (31, 35, 40), f_badge, pad=(8, 3))
        y += tile_h + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panels", nargs="+", default=None)
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--horizon", type=int, default=50, help="evaluation budget in env steps")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--gif-scale", type=float, default=0.6)
    ap.add_argument("--strip-steps", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50])
    args = ap.parse_args()
    scene = dict(np.load(args.scene))
    panels = args.panels or [str(x) for x in scene["labels"]]
    titles = dict(DEFAULT_TITLES)
    if args.titles:
        titles.update(zip(panels, args.titles))
    ep = Episode(scene, args.episode, panels, args.frames)
    args.out.mkdir(parents=True, exist_ok=True)

    steps = [t for t in ep.steps if t <= args.horizon]
    frames = [compose_frame(ep, titles, t, args.horizon) for t in steps]
    hold = int(round(1.5 * args.fps))
    frames = frames + [frames[-1]] * hold

    import imageio.v2 as imageio

    W, H = frames[0].size
    pad = lambda im: np.asarray(im.crop((0, 0, W + (-W) % 16, H + (-H) % 16)))  # noqa: E731  (libx264 wants /16)
    mp4 = args.out / f"ep{args.episode}_compare.mp4"
    imageio.mimwrite(mp4, [pad(f) for f in frames], fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
    small = [f.resize((int(W * args.gif_scale), int(H * args.gif_scale)), Image.LANCZOS) for f in frames]
    pal = [s.convert("P", palette=Image.ADAPTIVE, colors=192) for s in small]
    gif = args.out / f"ep{args.episode}_compare.gif"
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    strip = film_strip(ep, titles, args.strip_steps, args.horizon)
    strip.save(args.out / f"ep{args.episode}_strip.png", optimize=True)
    frames[len(steps) - 1].save(args.out / f"ep{args.episode}_final.png", optimize=True)
    print(f"[compose] {mp4}, {gif}, ep{args.episode}_strip.png ({len(steps)} steps)")


if __name__ == "__main__":
    main()

"""Render real expert/baseline/guided replay frames without changing task imagery."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 19)
    small = ImageFont.truetype(str(args.font), 14)
    for task in ('cube', 'pusht'):
        source = args.frames / task
        baseline = np.load(source / 'baseline.npz', allow_pickle=False)
        guided = np.load(source / 'guided.npz', allow_pickle=False)
        meta = json.loads((source / 'metadata.json').read_text())
        # Renderer frame 0 can already include the first method-specific action.
        # Shared initial task is checked against result manifests, not RGB equality.
        assert np.array_equal(baseline['dataset'], guided['dataset'])
        expert = guided['dataset']
        frames = []
        count = len(guided['agent'])
        for i in range(count):
            im = Image.new('RGB', (1040, 432), '#F6F7F9')
            draw = ImageDraw.Draw(im)
            panels = [expert[min(i, len(expert)-1)], baseline['agent'][i], guided['agent'][i]]
            labels = ['Expert demonstration', 'LeWM + original CEM', 'LeWM + action-prior-guided CEM']
            for j, (panel, label) in enumerate(zip(panels, labels)):
                x = 8 + j * 344
                draw.text((x, 10), label, font=font, fill='#15202B')
                im.paste(Image.fromarray(panel).resize((336, 336), Image.Resampling.LANCZOS), (x, 42))
                step = min(i, len(expert)-1) if j == 0 else i
                draw.rounded_rectangle((x+7, 49, x+80, 73), radius=6, fill='#202A35')
                draw.text((x+13, 52), f'frame {step}', font=small, fill='white')
            draw.text((8, 383), 'Expert ends at frame 25; final frame held', font=small, fill='#536273')
            draw.text((352, 383), 'Recorded outcome: failure', font=small, fill='#A23E35')
            draw.text((696, 383), 'Recorded outcome: success', font=small, fill='#177553')
            draw.rectangle((8, 408, 1032, 412), fill='#D5DAE1')
            draw.rectangle((8, 408, 8+int(1024*(i+1)/count), 412), fill='#4D7099')
            frames.append(im)
        gif = args.output / f'{task}_expert_comparison.gif'
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=200,
                       loop=0, optimize=False, disposal=2)
        # GIF is the README preview; MP4 preserves more colors and is smaller.
        import imageio.v2 as imageio
        imageio.mimwrite(args.output / f'{task}_expert_comparison.mp4',
                        [np.asarray(f) for f in frames], fps=5, codec='libx264', quality=8,
                        macro_block_size=8, ffmpeg_params=['-movflags', '+faststart'])
        frames[20].save(args.output / f'{task}_preview.png')
        print(task, meta['episode'], gif)


if __name__ == '__main__':
    main()

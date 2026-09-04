import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Save a model checkpoint every ``epoch_interval`` epochs.

    ``keep_last`` bounds disk usage by deleting older ``weights_epoch_*.pt``
    files, keeping only the most recent ``keep_last`` (plus the final epoch).
    Set ``keep_last=None`` to keep every checkpoint (old behaviour).
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1, keep_last: int | None = None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = max(1, int(epoch_interval))
        self.keep_last = keep_last

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        is_interval = epoch % self.epoch_interval == 0
        is_final = epoch == trainer.max_epochs
        if is_interval or is_final:
            self._save(pl_module.model, epoch)
            self._rotate()

    def _ckpt_dir(self):
        from stable_worldmodel.data import get_cache_dir

        return get_cache_dir(sub_folder='checkpoints') / self.run_name

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        try:
            save_pretrained(
                model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f'weights_epoch_{epoch}.pt',
            )
        except (RuntimeError, OSError) as e:
            # Most commonly a full disk. Free space by removing old checkpoints
            # and retry once so training is not lost.
            print(f"[SaveCkptCallback] save failed ({e}); pruning old checkpoints and retrying")
            self._rotate(force_keep=1)
            save_pretrained(
                model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f'weights_epoch_{epoch}.pt',
            )

    def _rotate(self, force_keep: int | None = None):
        keep = force_keep if force_keep is not None else self.keep_last
        if keep is None:
            return
        try:
            ckpt_dir = self._ckpt_dir()
            files = sorted(
                ckpt_dir.glob('weights_epoch_*.pt'),
                key=lambda p: int(p.stem.split('_')[-1]),
            )
            for old in files[:-keep] if keep > 0 else files:
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception as e:  # never let cleanup crash training
            print(f"[SaveCkptCallback] rotate skipped: {e}")
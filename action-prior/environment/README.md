# Environment snapshot

- `conda-linux-64.lock`: current explicit Conda package list, exported from the experiment environment.
- `pip-freeze-linux.txt`: current pip freeze, with torch/torchvision removed because the reproduction guide installs their CUDA 12.8 builds explicitly. The editable stable-worldmodel dependency is also installed separately at its recorded Git revision. Conda's local build-machine reference for `packaging` is replaced with the same recorded version, `packaging==26.3`.

The lock is an environment record, not proof of a fresh successful installation. It is Linux-specific. Upstream release availability and driver compatibility can change. Use the pinned source revisions from the reproduction guide, not just the package version number.

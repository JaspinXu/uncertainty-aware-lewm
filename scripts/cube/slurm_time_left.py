"""Print the minutes left in the current Slurm job (large number outside Slurm)."""

import os
import subprocess


def parse(s: str) -> float:
    s = s.strip()
    if not s or s.upper() in ("UNLIMITED", "NOT_SET", "INVALID"):
        return 1e6
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 1440 + h * 60 + m + sec / 60


def main() -> None:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        print(10**6)
        return
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True, timeout=30).stdout
        print(int(parse(out)))
    except Exception:
        print(170)


if __name__ == "__main__":
    main()

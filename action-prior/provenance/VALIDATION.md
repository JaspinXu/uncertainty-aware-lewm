# Packaging validation

Checked on 2026-10-24 (Asia/Singapore).

- Retrieved both final source directories from the experiment server. All 10 core files matched the SHA256 values in the final reports before portable root-path edits.
- Verified hashes for all published experiment JSON files and packaged source records.
- Recomputed the success means, sample standard deviations and average batched solve times from all 70 final test reports. All match their recorded summaries.
- Checked that each task uses matching 50-trajectory test manifests and the five expected planner seeds.
- Verified both selected demonstrations against final seed-62 records: original CEM fails and guided CEM succeeds. Their recordings were previously replayed with exact full-batch outcome agreement.
- Re-rendered 50-frame, 10-second GIFs and MP4s from the saved RGB arrays. Expert clips stop at frame 25 and are held afterward; no new robot action frames were synthesized.
- Parsed all Python source files for syntax errors and scanned text outputs for personal cluster roots and credential patterns. No matching personal root or credential pattern remained after replacing a nonportable Conda build-machine package reference.
- On a scheduled CPU node in the existing experiment environment: Cube 6/6 planner tests passed; PushT 8/8 passed. The tests cover original-CEM equivalence at zero guidance, unchanged later updates, first-iteration mixing, dimensions, gradients, image preprocessing and valid action windows. CPU-only NVML and unavailable ALE warnings were non-fatal.

Not performed: fresh environment installation, fresh GPU training/evaluation, or GitHub publication. No new performance result is claimed by this packaging work.

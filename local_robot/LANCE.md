# Local Lance training data

HDF5 remains the raw recording. This optional local path exports validated rows
to Lance and feeds the existing DiT trainer. Upstream code and Gontrol are
unchanged. Every command below is offline; none connects to a robot or camera.

## Setup

Install only extra packages into a separate directory inside `local_robot`:

```bash
uv pip install --python .venv/bin/python --target local_robot/.lance-deps \
  -r local_robot/requirements-lance.txt
```

Run from the repository root with `PYTHONPATH=local_robot/.lance-deps`.
The existing `.venv` supplies torch, OpenCV, HDF5 and the normal CLIP assets.
`pylance` is the file-format library; no database service is required.

## Export

```bash
PYTHONPATH=local_robot/.lance-deps .venv/bin/python -m local_robot.lance_export \
  --session local_robot/scripted_sessions/SESSION
```

Default output: `SESSION/lance_export`. It requires at least two validated
scripted recordings in `SESSION/recordings/*.h5` and rejects an existing output
directory. It never overwrites or deletes HDF5 inputs. A temporary export directory
is renamed into place only after all episodes and metadata are written.

Default image mode copies JPEG bytes unchanged. To match the existing MP4 export:

```bash
PYTHONPATH=local_robot/.lance-deps .venv/bin/python -m local_robot.lance_export \
  --session local_robot/scripted_sessions/SESSION \
  --output local_robot/scripted_sessions/SESSION/lance_224 --image-size 224 168
```

Resizing re-encodes JPEGs at quality 90. This trades space for resolution and is
recorded in the manifest. Native JPEGs remain available in the raw HDF5 files.

### Contents

`dataset.lance` contains the Gontrol core columns: `episode_id`, `task`,
`frame_index`, `timestamp_ns`, `image_top`, `image_left_wrist`,
`image_right_wrist`, `state` float32[14], and `action` float32[14]. Extra columns
are `split`, `usable`, and `source_timestamps_ns` (each camera/state/action stream).

`manifest.json` pins the Lance version and records source paths/SHA-256 hashes,
source attributes, validation results, image size, package versions, units,
absolute-action convention, and a fingerprint. `norm_stats.json` uses training
episodes only. `train/` and `val/` contain small compatibility metadata files.
The seed-123 80/20 episode split matches the MP4 exporter (8/2 for ten episodes).

Timing deliberately matches our existing HDF5 exporter: top-camera timestamps,
with each stream sampled causally at or before that timestamp. It does **not**
resample onto Gontrol's MCAP converter's exact 30 Hz grid. Sensor gaps/age above
300 ms and incomplete motion coverage are rejected. Missing fields are rejected;
there is no zero-fill fallback. Raw telemetry preserves unsynchronized samples.

This first adapter accepts the local scripted recorder's schema and endpoint
metadata. It is not a generic importer for arbitrary teleoperation HDF5 files.

## Train and resume

```bash
PYTHONPATH=local_robot/.lance-deps .venv/bin/python -u -m local_robot.lance_train \
  --dataset local_robot/scripted_sessions/SESSION/lance_export \
  --run-dir local_robot/scripted_sessions/SESSION/lance_run

PYTHONPATH=local_robot/.lance-deps .venv/bin/python -u -m local_robot.lance_train \
  --dataset local_robot/scripted_sessions/SESSION/lance_export \
  --run-dir local_robot/scripted_sessions/SESSION/lance_run --resume
```

The first command runs the existing small DiT test: 200 steps, hidden size 256,
depth 4, heads 8, batch 1, chunk 30, full randomly initialized DINO and normal
CLIP text conditioning. It saves at 100/200. `--resume` restores checkpoint 200
and runs/saves step 201. A new run needs a fresh `--run-dir`.

The adapter reads image columns for one timestep and scalar columns for its
30-step action chunk, entirely within the same episode. It reuses the original
preprocessing, prompts, sampler, optimizer, validation, and checkpoint writer.
DataLoader workers use `spawn` and open separate handles to the pinned version.
The dataset fingerprint is stored in both the run directory and checkpoints;
resume rejects a changed manifest, statistics, or dataset binding.

Keep exports immutable: make a new output directory for new conversion settings.
Appending Lance versions does not affect pinned readers, but do not replace
dataset files in place or remove old versions needed by a checkpoint. The binding
is a reproducibility check, not authentication against external file tampering.
Operator/subtask sidecars are not part of this scripted-data export.

The old `local_robot.scripted_train` MP4 path remains available.

## Benchmark

```bash
PYTHONPATH=local_robot/.lance-deps .venv/bin/python -m local_robot.lance_benchmark \
  --mp4-root local_robot/scripted_sessions/SESSION/export \
  --lance-root local_robot/scripted_sessions/SESSION/lance_224 \
  --output local_robot/scripted_sessions/SESSION/lance_benchmark.json
```

Uses matching episodes, scalar values, prompts, sample indices and resolution.
Warms every sampled index, alternates backend order across three rounds, and
reports median/p95 latency, samples/s, CPU time and training-data disk size.
MP4 and JPEG compression produce small pixel differences, reported separately.
Review MP4s remain in the original export; they are excluded from training size.
The resolution check covers the first sample: use uniform-resolution exports.

## Tests

```bash
PYTHONPATH=local_robot/.lance-deps .venv/bin/python -m pytest \
  local_robot/tests/test_lance_data.py local_robot/tests/test_lance_train.py \
  local_robot/tests/test_lance_benchmark.py -q
```

Tests generate their own data. PyTorch's spawned-worker test needs local IPC;
restrictive sandboxes can block it. It has a 30-second timeout.

## Recorded experiment

Existing synthetic session: `local_robot/scripted_sessions/20260928T015734Z/synthetic`.
See `lance_export/manifest.json`, `lance_run/train_audit.json`,
`lance_run/resume_audit.json`, `lance_benchmark.json`, and `lance_results.json`.
There are still no physical episodes. Synthetic flat-color images test the
pipeline; their performance and loss do not establish real-data performance
or policy quality. Benchmark representative recordings before making Lance the
default for larger training runs.

Measured on the synthetic fixture:

| Metric | MP4 + binary | Lance |
|---|---:|---:|
| Warm loader samples/s | 530.9 | 560.4 |
| Median sample latency | 1.83 ms | 1.69 ms |
| Training median logged steps/s | 13.81 | 13.60 |
| Training dataset size | 1.70 MB | 2.37 MB |
| Peak GPU allocation | 1.84 GiB | 1.84 GiB |

Training rate excludes the first interval and first interval after validation;
it is not total runtime including checkpoint writes. Both 200-step runs had finite
loss/gradients; the first/last 20-step mean losses were about 1.624/0.531.
Lance checkpoint reload and step 201 passed. Keep both formats available: the
roughly 6% loader improvement here did not translate into faster GPU training.

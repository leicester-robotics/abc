# Local Lance adapter

Intent: implement the approved HDF5 -> validated Lance -> existing trainer path
locally and simply. Preserve original source and raw HDF5. No robot/camera access.

1. Export scripted HDF5 files into Gontrol-compatible JPEG/state/action columns.
   Preserve camera timeline and causal sample selection for equivalence with the
   existing export. Store source timestamps, hashes, source metadata and validation.
   Fixed episode split (seed 123), training-only stats, atomic directory publication,
   pinned Lance version and schema/manifest. Native JPEG bytes by default; optional
   224x168 resize for the benchmark. Reject missing/nonfinite/stale/unusable data.
2. Add a chunk-aware Dataset adapter with per-worker snapshot handles. Read one
   frame's three images and 30 scalar action rows, never crossing episode boundaries.
   Reuse upstream image preprocessing, prompt handling, samplers and trainer.
   Keep the original MP4 loader available. Bind resume to the manifest fingerprint.
3. Add a small CLI for Lance training and a CPU loader/disk benchmark. Run on the
   existing ten synthetic episodes only. Compare identical frames and resolution;
   JPEG and MP4 pixels may differ from compression. Verify scalar and prompt parity,
   episode boundaries, byte preservation, validation failures, snapshot pinning,
   split/stats isolation, checkpoint reload, and finite optimizer updates.
4. Run 200 steps and one resumed step with Lance, compare with a fresh MP4 run under
   the same configuration. Report cache/storage and synthetic-data limitations.

Implementation stays in this existing feature checkout under local_robot, as
explicitly requested. Existing HDF5, MP4 export and trainer behavior are preserved.
Proceed inline using the approved design; no additional design approval needed.

Progress: implementation, offline training, comparison, and verification complete.

Completed:
- Dependencies isolated under local_robot/.lance-deps, exact primary versions in
  requirements-lance.txt; neither existing Python environment modified.
- Export and adapter tests passed, including unchanged JPEG bytes, causal values,
  unusable/nonfinite rejection, train-only normalization and pinned snapshots.
- Spawned DataLoader passed outside sandbox. Sandboxed torch IPC hung; cancelled
  that test and added a 30-second worker timeout.
- Exported the existing ten synthetic episodes at 224x168 into 1,520 Lance rows,
  seed-123 split 8/2. Original raw recordings and MP4 export remain intact.
- Lance trainer completed 200 finite optimizer steps, saved 100/200 checkpoints,
  reloaded 200 and completed/saved step 201 with matching dataset binding.
- Fresh MP4 comparison completed 200 steps in a separate run directory.
- Independent review: fixed warm-cache benchmark by warming every sampled index;
  regression failed before the fix and passed afterward. Added boundary/projection,
  missing/stale/corrupt stream and manifest-tamper regressions.

Results (synthetic; single local comparison): warm CPU samples/s MP4 530.88,
Lance 560.43; median latency MP4 1.828 ms, Lance 1.691 ms. Training median logged
steps/s MP4 13.81, Lance 13.60 (excludes first interval and first after validation).
Training data bytes MP4 1,698,000; Lance 2,371,888. Both peak 1.84 GiB GPU.
Do not generalize these numbers to real video, cold caches, or remote storage.

Ruling: preserve current top-camera timeline rather than changing to exact 30Hz
resampling; this gives scalar/frame parity with existing exports. Consequence:
Gontrol-compatible row schema does not imply identical MCAP synchronization.
Ruling: retain both loaders; modest synthetic read gains did not improve training
throughput. Broader data is needed before changing a default.

Deferred review limitations: resolution parity checks the first sample only, so
benchmark uniform-resolution exports. Version/row count/fingerprint assume the
published dataset files remain intact, not replacement or malicious modification.
These are documented in LANCE.md; no hardware validation was attempted.

Final verification: 36 data/training/camera-fixture tests passed in the training
environment (including spawned-worker IPC), and 32 existing lift tests passed in
the hardware SDK environment without opening hardware. Total: 68 passed, no skips.
The four new modules compile; git diff confirms upstream deploy, abc_minimal and
pyproject.toml are unchanged. All new work remains local and uncommitted.

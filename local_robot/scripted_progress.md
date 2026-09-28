# Scripted collection and DiT test

Source: user-supplied plan, 2026-09-27.

Tasks: (1) collector and hardware-free tests, (2) synthetic export,
(3) first physical episode validation and nine further cycles,
(4) export 8/2 and train 200 steps, reload and train one step.

Interfaces: recorder schema -> existing exporter -> existing DiT loader.
All use top/left/right cameras and left 7 + right 7 absolute joint actions.

Ruling: work in the current feature checkout, with additions under local_robot,
as requested; no upstream files or existing local implementation changed.
Ruling: reuse local lift Session for measured-pose initialization and shutdown;
continuous local sampling replaces ZMQ followers. This avoids blocking reset
and provides SDK feedback-age checks. No leader or policy processes are needed.

Progress: baseline inspection complete. CAN interfaces UP; GPU free.

Collector: initial 42 hardware-free tests passed, including four review fixes
(repeat interrupts, hold/command race, abort usability, worker shutdown order).
Synthetic: ten generated HDF5 episodes validated and exported as 8 train / 2 val.
Training: 200 synthetic optimizer steps passed; checkpoint 200 reloaded and
step 201 passed. Full DINO architecture, random initialization, normal CLIP text.

Physical attempt: stopped at sensor startup before any episode or scripted leg.
Right motor 7 and left motor 6 reported communication loss; SDK workers stopped.
User confirmed both arms supported and authorized recovery `disable` input.
All motors returned fault status 0xD rather than disabled acknowledgments;
operator was told to switch off motor power. No successful shutdown claim made.

Camera-only diagnostic: native camera operations paused a Python heartbeat for
0.377 seconds. Ruling: isolate cameras in spawned processes, wait for readiness
before enabling motors; removes this source of GIL stalls, but hardware fix is
unverified. Motor communication faults may have additional causes.

Latest user instruction: robot is now in use by someone else; do only internal
changes. No further hardware commands. Physical collection/training deferred.
Ruling: latest request for smooth return supersedes constant-speed timing.
Use quintic time scaling on the same straight joint-space path in both directions,
with zero endpoint velocity/acceleration, <=0.25 rad/s and <=0.5 rad/s².
Faults retain hold-for-recovery behavior; normal return targets the saved start.

Normalization: existing exporter computes stats before splitting. Local wrapper
recomputes stats from eight training episodes only, avoiding validation leakage.

Final independent review: fixed both important findings with failing-then-passing
tests: truncated/shifted camera coverage and complete-but-unusable export input.
Validation now checks common stream coverage and synchronized sample age; export
asserts exact counts, no skipped episodes/segments, and the requested split.

Deferred review minors: additional camera-process stress tests (stale/dead worker
during recording, queue saturation, forced termination); the camera worker comment
mentions sequence gaps although only timestamp gaps are validated. No sequence
numbers are currently persisted. These do not change the documented 300 ms
freshness threshold; isolated camera behavior still needs physical validation.

Final review exclusions: physical tracking, collision clearance, CAN reliability,
and the hardware effect of camera process isolation remain unverified because the
user prohibited further hardware work. Do not infer hardware readiness from the
software tests. No policy was deployed.

Broad pytest discovery also entered installed h5py's own tests: 928 passed,
18 skipped, four MPI fixture errors (test_mpio, test_mpio_append, test_mpi_atomic,
test_close_multiple_mpio_driver; mpi_file_name fixture unavailable). Project tests
are run with --ignore=local_robot/hardware_deps; dependency tests are not repaired.
Torch audit tests run separately in .venv because Joint Studio has no torch.

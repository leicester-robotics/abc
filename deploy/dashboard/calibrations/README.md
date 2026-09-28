# Station calibration provenance

Copied unchanged from `origin/mhasek/abc_gello` commit `d4abb33`.
The mapping implementation in `../saved_calibration.py` comes from the same
commit's `deploy/gello/control/calibration.py`.

The JSON side labels are reversed relative to the operator-confirmed station:

- `left.relative.json`: adapter Y3, physically **right**.
- `right.json`: adapter Y6, physically **left**.

The dashboard binds calibration to adapter path, servo IDs, and baudrate.
`configs/station.json` sets the actual side. Saved JSON files remain unchanged.
The measured joint envelope is retained for provenance; simulation uses the
MuJoCo actuator limits. This envelope is not a physical-control authorization.

# Experiment index

This directory holds runnable scripts and input configurations. Results belong
beside the corresponding outputs as `output/<run>/RESULTS.md`.
Only `RESULTS.md` and `TF_PROTOCOL.md` are versioned under `output/`; model weights,
volumes, images, logs and generated metric JSON files remain ignored.
Historical multi-run reports are kept at the common output parent to preserve
their comparisons without duplicating or discarding results.

| Experiment | Report |
|---|---|
| Static reproduction runs (historical aggregate) | [Results](../output/experiments/RESULTS.md) |
| vorts0001 static baseline | [Results](../output/experiments/vorts0001_baseline/RESULTS.md) |
| Initial dynamic integration checks (aggregate) | [Results](../output/dynamic/RESULTS.md) |
| Full 90-frame geometry-deformation training | [Results](../output/dynamic/vorts90_full_geometry/RESULTS.md) |
| Official distributed TF, offline reevaluation | [Results](../output/dynamic/vorts90_full_geometry/official_tf_metrics/RESULTS.md) |
| Preserved custom-TF evaluation protocol | [TF protocol](../output/dynamic/vorts90_full_geometry/supplemental_metrics/TF_PROTOCOL.md) |

The custom-TF results and official-LUT offline results are separate protocols;
neither should be mislabeled as full native paper-renderer parity.

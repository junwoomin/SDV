# Preserved earlier SDV implementation

[Current repository map](../docs/REPOSITORY_LAYOUT.md)

This folder preserves supplied source and small fixtures. It is not selected by the current Studio job path merely because it is present here.

| Path | Evidence and use |
| --- | --- |
| `carla_data_gan/` | Earlier collector/navigation/rendering source; current workflow imports `sdv.collection.carla_engine` instead. The top-level `carla_data_gan` package keeps old import names resolvable. |
| `training/modeling/` | Earlier duplicated model source that the supplied `train_model/__init__.py` already bypassed. Current training imports `sdv.training.modeling`. The top-level `train_model` namespace continues to resolve to the current implementation for external scripts/checkpoints. |
| `utils.py` | Historical mixed UI/CARLA/training helpers. Current code uses dedicated `sdv` helpers; top-level `utils` aliases this module for external imports. |
| `fixtures/test_img/` | Original small desktop sample images; `sdv.ui.workflow` paths point here. They are not measured predictions of the current demonstration. |

All original notices and source bytes are retained except documented relative asset paths in the active desktop UI. These sources retain their historical dependency assumptions and are not advertised as validated standalone tools. The missing `experiments.training.detection` target of `trainx2.py` was already absent in main; no experiment was fabricated.

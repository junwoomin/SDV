# Development notes

[한국어](development_ko.md) · [Project overview](../README.md)

This is a source release of an unfinished research prototype. Expect unresolved errors. Syntax checking is the only execution-related check performed for this publication. The CARLA server, CUDA training, desktop UI, and device deployment were not run here.

## Verified source-level blockers

| Area | Observation | Work needed |
| --- | --- | --- |
| UI training launch | `start.py` launches `python train.py`, but that file is absent from the supplied source | Decide the intended trainer and connect it explicitly |
| Training configuration | The UI writes `data/loop/train_config_*.yaml`; both trainers read `data/train_config.yaml` | Define a common configuration contract and select the intended experiment |
| Training parameters | Both trainers hardcode epochs, batch size, and optimizer settings | Connect the parameters selected in the UI to the trainer |
| CARLA versions | `requirements.txt` and the old installation notes referenced 0.9.13; `start.py` installs/downloads 0.9.15 | Choose and verify one matching server/Python API version |
| Dependencies | `requirements.txt` does not cover imports such as Torch, Torchvision, PyYAML, Pillow, Pandas, Matplotlib, SciPy, PyQuaternion, EfficientNet-PyTorch, Torch Scatter, and Webcolors | Reconstruct and test an environment before treating it as reproducible |
| Saved test input | `train2.py` defaults to `img/test.pth`; the other trainer uses `img/test.pt` through `utils.py` | Recreate compatible test tensors and agree on a path. Saved tensors are excluded from this release |
| Dataset inputs | The loader expects generated `data/train/`, `data/val/`, and `data/sensor_config.yaml` | Generate compatible data and verify annotations, calibration, and splits |
| Prediction/display | Result views depend on generated logs, images, and prediction folders | Verify the output schema against the UI |

This is not an exhaustive bug list. A passing syntax check cannot detect missing packages, tensor shape errors, invalid labels, CARLA compatibility problems, or training failures.

## Environment assumptions

The source targets a Linux desktop, Pygame, CARLA, and PyTorch/CUDA training. It expects the CARLA server launcher at `carla/CarlaUE4.sh` under the repository root. The UI uses the NanumGothic font at `/usr/share/fonts/truetype/nanum/NanumGothic.ttf`.

`start.py` contains automatic package installation, system-package installation, CARLA download/extraction, and CARLA process cleanup. Review that bootstrap before launching it in an existing environment. Those operations were not executed during publication.

The historical dependency list is retained for reference. Do not assume that installing it alone produces a working runtime.

## Manual bring-up targets

After resolving the version and configuration issues, the intended entry points are:

```bash
# Desktop UI
python start.py

# Data collector; requires data/sensor_config.yaml and a local CARLA installation
python carla_run.py

# Experimental trainers; require a compatible config, dataset, and saved test input
python train2.py
python trainx2.py
```

These commands identify the scripts to investigate. They are not a tested quick-start procedure. `sensor_config.yaml` at the repository root is an example and is not automatically the configuration read by the collector.

## Public source cleanup

The release retains the current UI, CARLA collection path, training models/scripts, navigation code imported by that path, and small UI/sample assets.

Excluded files include Python bytecode/caches, `carla_data_gan/main (Copy).py`, `main22.py`, the unreferenced `provider.py`, the disconnected `agents_map/` tree, the standalone unused CARLA launcher, the disconnected `models/` LightFormer experiment and its local-path configuration, the scratch notebook, obsolete installation notes, and the large saved test tensor. The two training scripts remain because they contain different experiments.

An unused helper that killed unrelated CUDA Python processes was removed from `utils.py`. Core model and collection behavior was otherwise preserved rather than repaired in this publication.

The inspected publishable files contain no detected hardcoded API keys, access tokens, passwords, or private keys. `.gitignore` excludes common credential files, local environments, simulator installs, generated data, checkpoints, and experiment output. It does not replace review of future commits.

## Third-party notices

CARLA-derived navigation modules retain their original MIT copyright and license headers. `carla_data_gan/map_utils.py` retains its CARLA source attribution. The project [MIT license](../LICENSE) remains unchanged.

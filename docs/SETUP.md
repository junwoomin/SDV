# Environment and separately supplied assets

[Research prototype overview](../README.md) · [한국어](SETUP_KO.md) · [Illustrated workflow](USER_GUIDE.md)

## Web console

On Linux with Python 3.7+, run from the repository:

```bash
./start.sh
# Select another web interpreter explicitly, when needed:
SDV_WEB_PYTHON=/path/to/python ./start.sh
```

Open `http://127.0.0.1:8765`. The launcher prepares a separate `.venv-web` (or `.venv-web-pyXY`) and installs `requirements/web.txt`. It preserves an existing environment built with another Python. Settings selects the independent research interpreter; do not install the web dependency set over an existing research environment. A new web environment requires network access.

## Research environment and CARLA

Reuse your existing validated Python/CARLA/PyTorch environment if available. Otherwise prepare a separate environment compatible with the GPU, PyTorch/torchvision and CARLA Python wheel before installing [research dependencies](../requirements/research.txt). Review these dependencies against your selected CUDA/PyTorch and CARLA environment. [Roach dependencies](../requirements/roach.txt) describe the dedicated Python 3.10 collector environment. Wheel availability must match your selected Python and server.

In Settings select the research Python executable and existing CARLA engine directory. Keep **CARLA server and API 0.9.15**, distinct RPC/TM ports, and **Epic + Off-screen**. Run diagnostics. If no engine is installed, Studio's **Install & start CARLA** prepares the matching client in the selected interpreter and downloads the engine after your action. The engine is about 7.81 GiB; initial free-space requirement is 45 GiB. Existing/broken mismatched clients are reported rather than overwritten. An independently started server may be connected separately.

## Asset checklist

| Asset | Expected location / selection | Verify |
| --- | --- | --- |
| CARLA engine | `carla/CarlaUE4.sh`, or existing absolute installation path in Settings | Version, executable, RPC Ready and actual world snapshot |
| Research Python | Settings → Python executable | CARLA/PyTorch/torchvision and required research imports in this interpreter |
| Native map package | `assets/bev_maps/catalog.json` plus matching `data5/` and `data10/` town files; or a project-local Map Root | Catalog, file hashes, offsets/resolution and selected Town availability |
| Roach checkpoint | `roach/log/ckpt_11833344.pth`, or your project-local checkpoint selection | Compatible policy observation contract and model-load check |
| Training dataset | `data/train`, `data/val`, or a prepared dataset pair below `data/` | Scene separation, aligned IDs, target availability and original calibration |
| Calibration | Recorded `sensor_config.yaml` copied with scenes/dataset | Actual camera geometry agrees with labels and training config |
| Encoder weights | `.runtime/cache/torch/hub/checkpoints/` | Training **Check readiness** downloads only the selected official file and checks its hash |

Supply map **catalog metadata together with the exact map files**; copying PNG folders alone is insufficient. The repository includes no map catalog or distributable checkpoint. Do not invent missing maps, calibration, road-height images or labels. Asset redistribution rights are separate from the code license. See [BEV map requirements](GLOBAL_MAP_INTEGRATION_KO.md).

Dataset preparation copies scene data and its recorded calibration without changing the source. Generated collection data, training outputs, checkpoints, environments and `.runtime` records are ignored by Git. An optional old `img/test.pth` / `img/test.pt` may still be supplied by its owner; training can instead preview real validation samples. Desktop example images live in `sdv/ui/assets/samples/` and do not replace a training dataset.

## Before each actual run

Save settings, stop sensor preview, check CARLA Ready and collection readiness. For training select the complete TRAIN/VAL dataset, tasks, calibration and official weights, then pass training readiness. Do not infer runtime readiness from unit tests. See [the visual guide](USER_GUIDE.md), [prototype status](../README.md#project-status-and-requirements) and [development checks](development.md).

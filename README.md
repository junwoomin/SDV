# SDV Studio

![SDV Studio English dashboard](docs/images/dashboard-en.png)

*Main dashboard for collection, training and experiment management.*

**SDV Studio is a research prototype for autonomous driving experiments in CARLA.** It brings simulation setup, sensor collection, dataset preparation, perception training and result inspection into a local web workspace. The interface supports Korean and English through the **KO / EN** selector.

[한국어](README_ko.md) · [Illustrated user guide](docs/USER_GUIDE.md) · [Setup](docs/SETUP.md) · [Repository structure](docs/REPOSITORY_LAYOUT.md)

## Features

| Area | What it does |
| --- | --- |
| Simulation | Configure CARLA, towns, traffic, driving modes and engine diagnostics; connect a compatible Roach policy. |
| Sensors and collection | Configure camera/LiDAR/IMU inputs, preview sensors, collect aligned samples and inspect saved progress. |
| Sequence inspection | Replay saved RGB, segmentation, depth, object annotations and BEV by frame. |
| Dataset preparation | Split complete scenes into TRAIN/VAL and copy data with its recorded calibration. |
| Perception training | Select image encoders, image/BEV decoders, tasks, losses and training parameters. |
| Experiment management | Queue jobs, pause/resume at epoch boundaries, inspect metrics/checkpoints and browse saved predictions. |

## Start

On Linux with Python 3.7 or later:

```bash
./start.sh
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The launcher prepares a separate web environment and installs its dependencies. In **Settings**, select the research Python interpreter and CARLA 0.9.15 installation used for collection and training. [Setup](docs/SETUP.md) explains the required environments and assets.

## Workflow

1. **Settings:** save the research interpreter, CARLA installation and storage paths; run diagnostics.
2. **Simulation / Sensors:** choose towns, driving mode and sensor outputs; start CARLA and check readiness.
3. **Collection:** collect samples and inspect saved previews, counts and job logs.
4. **Sequences / Dataset:** replay data, assign separate scenes to TRAIN/VAL and prepare the dataset.
5. **Training:** select models, tasks, losses and parameters; check readiness and start or queue a job.
6. **Monitoring / Results:** inspect metrics, checkpoints and saved Input / Ground truth / Prediction images.

[Follow the workflow with screenshots →](docs/USER_GUIDE.md)

## Project status and requirements

This prototype provides a workspace for developing and inspecting research experiments. Driving-policy transfer, BEV camera geometry, long-running stability and model convergence require further evaluation. Screenshots illustrate the interface; model quality must be assessed on an appropriate independent validation dataset.

CARLA, its matching Python API, research PyTorch/CUDA dependencies, datasets and calibration are supplied separately. Global BEV maps require matching catalog metadata; Roach collection additionally requires a compatible checkpoint. Required assets are checked before the relevant job starts. Map and checkpoint redistribution rights are separate from the code license.

## Source structure

| Path | Role |
| --- | --- |
| `studio/` | Web interface, API, configuration and job management |
| `sdv/` | Collection, BEV generation, driving, training and desktop interface |
| `configs/`, `requirements/` | Configuration examples, scenarios and dependency groups |
| `scripts/`, `tests/` | Validation tools and regression checks |
| `docs/` | Setup, illustrated usage and technical documentation |

[Entrypoints and package details](docs/REPOSITORY_LAYOUT.md) · [Development checks](docs/development.md)

Code: [MIT license](LICENSE). Source attribution: [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Bundled Noto Sans KR fonts retain their SIL Open Font License.

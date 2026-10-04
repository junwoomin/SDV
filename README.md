![SDV multimodal data preview](docs/images/data-preview.png)

# SDV

[한국어](README_ko.md)

SDV is an early prototype of an autonomous-driving research platform, initiated at the suggestion of a senior principal researcher during my time at KETI and developed independently over approximately three months. The goal was to integrate the available FMTC CARLA map into a single UI covering sensor configuration, data generation, model training, and result inspection. It was not a formal research project, and development stopped when I left KETI, leaving some features and integrations unfinished.

SDV is an early autonomous-driving research workspace that connects CARLA sensor configuration, multimodal data collection, model configuration, training, and result inspection through a desktop UI. The aim is to make perception experiments easier to set up and compare.

> **Early research prototype.** This public snapshot still contains many unresolved errors and incomplete integrations. It may fail to launch, collect data, train, or display predictions without changes. End-to-end execution has not been verified for this release.

**Language support:** The current application version supports Korean only.

## Demonstrations

- [SDV UI demonstration](https://youtu.be/Y6UTaquOltg)
- [FMTC map: four-LiDAR fusion demonstration](https://youtu.be/UsxVp50MeAM)

The videos and screenshots show earlier demonstrations. They do not establish that this source snapshot reproduces every demonstrated feature. Numbers visible in the UI are examples from those sessions, rather than validated benchmark results for this release.

## Original workflow

The initial design puts the full experiment cycle in one interface:

1. Select CARLA towns, cameras, and LiDAR.
2. Set sensor position, orientation, image size, and field of view.
3. Generate and inspect RGB, segmentation, depth, object annotations, and BEV data.
4. Choose model components and perception tasks, then set training parameters.
5. Queue experiments, inspect training progress and predictions, and compare saved results before retraining.

| Area | Components in this snapshot | Current limitation |
| --- | --- | --- |
| Desktop interface | Pygame screens for sensor setup, data previews, model selection, training queues, and results | UI actions and backend entry points are not fully connected |
| CARLA collection | Sensor wrappers, navigation helpers, scene creation, and multimodal output in `carla_run.py` | Requires a compatible CARLA server and local setup |
| Perception models | EfficientNet/ResNet image encoders, BEV projection, FPN/BiFPN, optional PointPillar features, and task heads | Training and task combinations need debugging |
| Training experiments | `train2.py` and the object-detection variant `trainx2.py` | Missing UI training entry point, config mismatches, and hardcoded settings |
| Results | Local logs, metrics, and prediction display code | Requires compatible generated outputs and saved tensors |

### Sensor configuration

![Sensor placement and camera preview](docs/images/sensor-settings.png)

### Model and training configuration

![Training parameters](docs/images/training-config.png)

![Experiment queue](docs/images/training-queue.png)

### Training inspection

![BEV predictions during a previous training session](docs/images/training-preview.png)

## Planned research

The next design extends SDV from a CARLA experiment UI into a workspace for simulation, synthetic data, evaluation, and deployment. These are research directions and are not implemented as a complete pipeline in this release.

| Direction | Intended work |
| --- | --- |
| Lightweight 2D BEV simulation | Development of the lightweight BEV simulator was being pursued in the separate [LRS (Low Resource Simulation)](https://github.com/junwoomin/LRS) project, using an Ackermann kinematic model to study driving behavior and scenario generation with lower simulation overhead |
| CARLA validation | Validate selected LRS-generated scenarios in CARLA and explore reducing dependence on long-running CARLA data collection while making the research workspace more useful |
| Style transfer and data augmentation | Inspired by UniScene, the intended extension was to incorporate style transfer and data augmentation reflecting real data or the visual styles of other simulators, enabling training of perception models robust across diverse environments |
| BEV-conditioned generation | Investigate generating camera images/video, 3D occupancy, and LiDAR from BEV scenes, with controllable scene variations |
| Dataset evaluation | Compare perception performance on CARLA and nuScenes and measure the usefulness of generated training data |
| W&B integration | Track and analyze experiments, search hyperparameters, and save/share results through Weights & Biases |
| DEEPX and SDV-RUN | Investigate exporting compatible models to DEEPX and connecting inference to a separate runtime for prediction and control |

Generated data realism, geometric consistency, temporal consistency, and transfer to real data remain questions to test. This repository does not claim that synthetic data is equivalent to real data or that the planned deployment path already works.

The intended prediction interface also includes image and webcam inputs. A complete standalone prediction workflow remains unfinished.

[Research plan and proposed architecture](docs/research-plan.md)

## Source layout

| Path | Role |
| --- | --- |
| `start.py` | Desktop UI and experiment configuration |
| `carla_run.py` | CARLA startup and scene/sensor data collection |
| `carla_data_gan/` | CARLA world, sensor, navigation, and BEV rendering helpers |
| `train_model/` | Dataset loading, model components, voxel/pillar processing, and losses |
| `train2.py`, `trainx2.py` | Experimental training scripts |
| `utils.py` | Visualization, data, and process helpers |
| `sensor_config.yaml` | Example sensor and town settings |
| `img/` | Small UI assets and sample sensor data |
| `docs/` | Research plan, development notes, and screenshots |

## Working with the snapshot

Read [development notes](docs/development.md) before attempting to run it. The historical `requirements.txt` is incomplete and conflicts with some versions in the UI bootstrap. A working environment has not been reconstructed or tested here.

Known blockers include a missing `train.py` called by the UI, different configuration paths between the UI and training scripts, inconsistent CARLA versions, and saved test tensors that are not distributed. Python syntax checks pass, but they do not verify imports or runtime behavior.

Generated caches, duplicate scripts, disconnected legacy modules, notebooks, and large saved test tensors are excluded from the public source. Small assets used by the UI remain included. No hardcoded API keys, access tokens, passwords, or private keys were found in the inspected publishable files.

## License

[MIT](LICENSE). Existing CARLA copyright and license notices remain in the navigation code.

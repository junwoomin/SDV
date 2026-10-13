# Development checks

[한국어](development_ko.md) · [Overview](../README.md) · [Repository paths](REPOSITORY_LAYOUT.md)

The current implementation is `studio` plus `sdv`; historical publication assumptions are preserved in [original development notes](legacy/development_original_EN.md). Read [setup](SETUP.md) for the separate research environment and assets.

Use a dedicated test environment with `requirements/dev.txt`. Existing prepared environments may be reused without changing research packages:

```bash
python -m pytest -q
bash -n start.sh start-desktop.sh
python -m compileall -q sdv studio legacy train_model carla_data_gan
python -m studio --help
python -m sdv.collection.runner --help
```

When unrelated system pytest plugins (such as ROS launch testing) enter an incompatible interpreter, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for this repository's checks. The suite does not require those external plugins. Use a host that permits localhost sockets/subprocess communication for process/API tests.

Native-map integration tests need the separately supplied `assets/bev_maps/catalog.json` and exact map package. They skip with a reason in a source-only checkout. Other optional tests report missing native assets/dependencies or comparison sources. Supply originals to execute them; do not replace them with invented fixtures. This candidate's validation reports distinguish source-only and original-asset runs.

Tests and CPU checks do not establish real CARLA/CUDA correctness. The [actual execution records](SCREENSHOT_EVIDENCE.md), [engine validation](CARLA_STABILITY_KO.md) and [training validation](TRAINING_UI_FOLLOWUP_KO.md) retain their distinct scopes. Full simulator downloads, training, driver changes and remote publication are not part of a documentation cleanup.

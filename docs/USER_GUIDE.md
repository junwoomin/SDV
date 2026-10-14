# Use SDV Studio, step by step

[Overview](../README.md) · [한국어](USER_GUIDE_KO.md) · [Setup](SETUP.md)

SDV Studio is a CARLA research prototype. Screenshots illustrate the interface; paths, settings and values should be chosen for your own environment and dataset. Use the **KO / EN** selector to change the UI language.

## 1. Open the workspace and set the environment

Run `./start.sh`, open `http://127.0.0.1:8765`, and select **Settings** in the left menu. Enter the CARLA installation and research Python paths, check host/ports and data/result paths, then **Save system settings**. Use **System Diagnostics** to check the actual GPU/runtime. Keep **Off-screen · Epic**; changing the profile takes effect on the next engine launch.

![Actual settings screen with the existing CARLA path, ports and Epic off-screen profile](images/walkthrough/settings.png)

*Set the CARLA and research-interpreter paths for your machine.*

## 2. Choose CARLA and collection settings

Open **Simulation → Collection settings**. Choose towns, samples, camera inputs and the driving mode. For Roach, provide your compatible checkpoint and its required maps, then verify the model load. Save the settings. **Start CARLA** reuses an existing installation; **Install & start** is for a missing installation. Wait for **Ready**, then **Check readiness**. Fix each reported missing requirement before starting.

![Actual collection setup selecting Town01 and Town03 with a prepared CARLA engine](images/walkthrough/collection-settings.png)

*Choose towns and saved sample counts independently of the simulation tick rate.*

Open **Sensors**, enable the required outputs and enter position, rotation, resolution and field of view. Use live preview when needed; **Apply to preview** checks a draft, while **Save sensor settings** commits it for collection. Stop the sensor preview before collection.

![Actual sensor settings for the front camera and selected RGB, segmentation, depth and object annotations](images/walkthrough/sensor-settings.png)

*Numeric position, rotation, resolution and field of view define the recorded calibration.*

## 3. Collect and verify saved data

Return to collection, click **Start collection**, review the job snapshot and confirm. Watch actual saved-sample progress and saved RGB/BEV previews. When completed, check the final sample count, job state and logs. If a job fails, retain its outputs and inspect the reported engine/job error.

![Real running collection job with saved camera preview](images/walkthrough/collection-running.png)

![Completed collection and saved BEV preview](images/walkthrough/collection-complete.png)

*Check completion, saved-sample counts and logs before preparing a dataset.*

## 4. Split scenes, choose a dataset and replay it

Open **Sequence analysis / Dataset**, assign complete scenes to TRAIN and VAL, give the new dataset a name, and select **Prepare dataset**. Preparation copies data and recorded calibration; it preserves the source. Use different scenes rather than mixing neighboring frames randomly. Check the resulting split counts.

![Actual scene selection preparing a dataset with separate TRAIN and VAL scenes](images/walkthrough/dataset-split.png)

*Use separate scenes for TRAIN and VAL and preserve the source data.*

Choose a sequence and camera, then press Play or move the Frame slider. Verify aligned RGB, segmentation, depth, object annotations and BEV for the selected Frame ID. Select another BEV layer only when supplied; heed the yellow asset/height warnings.

![Actual synchronized stored-frame playback with RGB, segmentation, depth, object boxes and BEV](images/walkthrough/sequence-playback.png)

*Inspect actual annotations and available BEV layers; missing road height remains N/A.*

## 5. Select model, tasks, losses and training parameters

Open **Training → Setup** and select the prepared dataset from the TRAIN/VAL pair dropdown. Verify calibration and available inputs/targets. Select encoder and decoders, then enable only tasks supported by the data. Disabled tasks explain what is missing; readiness can still be incomplete at this stage.

![Actual model/task setup before the final training readiness check](images/walkthrough/model-tasks.png)

In **Task settings**, inspect observed segmentation classes and validity masks. Choose a supported loss for each enabled task and positive task weights; review the recommendation before applying class changes. Set epochs, batch size, learning rate, optimizer, input size and prediction-recording intervals. Save and run **Check readiness**; it verifies data, dependencies, CUDA and the selected official pretrained weights. A status read alone does not download weights. Start or enqueue only after readiness succeeds.

![Training parameters and readiness controls](images/walkthrough/training-config.png)

*Choose training parameters for the task and dataset, then run readiness checks.*

## 6. Watch training and use checkpoints

Select **Monitoring** to inspect actual epoch/step progress, measured ETA, logs and saved metrics. Queueing retains immutable configuration/data/calibration snapshots. **Pause** finishes the current epoch; **Resume** continues from its checkpoint. **Start again** creates a new run from epoch 1. Stopped-job continuation requires a compatible completed checkpoint.

![Training progress and checkpoint monitor](images/walkthrough/training-monitor.png)

*Inspect epoch/step progress and the checkpoint state before continuing a job.*

## 7. Browse predictions and interpret results

In Monitoring or **Results**, select an experiment, split, sample and task. Inspect **Input / Ground truth / Prediction** side by side. Move the Epoch or Iteration slider through saved positions only; missing positions are not synthesized. Validation samples are fixed per dataset content; comparison across experiments requires matching fingerprints.

![Real prediction browser showing validation input, segmentation ground truth and prediction](images/walkthrough/prediction-browser.png)

Scroll to loss graphs and task tables. Read the **TRAIN / VAL** legend and metric direction; Best records its epoch. Check saved configuration, `train_metrics.csv`, prediction history and best/last checkpoints in the experiment results. Queue deletion retains results.

![Actual TRAIN and VAL loss legends and per-task recorded metrics](images/walkthrough/metrics.png)

*Interpret metrics against the validation split and available labels. Screen values are examples.*

See [model and training settings](MODELS_KO.md), [BEV contracts](GLOBAL_MAP_INTEGRATION_KO.md) and [setup](SETUP.md) for supported options and prototype limitations.

# Use SDV Studio, step by step

[Overview](../README.md) · [한국어](USER_GUIDE_KO.md) · [Setup](SETUP.md)

Screenshots are unmodified browser captures of the verified 13 October run, using the same UI source as main `9fb7de2`. They show Korean labels; English equivalents below match the app's EN selector. The shown dataset and results belong to that run; your counts depend on your own data. [Provenance and limits](SCREENSHOT_EVIDENCE.md).

## 1. Open the workspace and set the environment

Run `./start.sh`, open `http://127.0.0.1:8765`, and select **Settings** in the left menu. Enter the CARLA installation and research Python paths, check host/ports and data/result paths, then **Save system settings**. Use **System Diagnostics** to check the actual GPU/runtime. Keep **Off-screen · Epic**; changing the profile takes effect on the next engine launch.

![Actual settings screen with the existing CARLA path, ports and Epic off-screen profile](images/walkthrough/settings.png)

*The demonstrated machine reused its existing CARLA and research interpreter. Paths in this image are examples from that PC, not paths to copy onto another computer.*

## 2. Choose CARLA and collection settings

Open **Simulation → Collection settings**. Choose towns, samples, camera inputs and the driving mode. For Roach, provide your compatible checkpoint and its required maps, then verify the model load. Save the settings. **Start CARLA** reuses an existing installation; **Install & start** is for a missing installation. Wait for **Ready**, then **Check readiness**. Fix each reported missing requirement before starting.

![Actual collection setup selecting Town01 and Town03 with a prepared CARLA engine](images/walkthrough/collection-settings.png)

*The demo selected Town01 and Town03 with 40 saved samples per town. The simulation ran at 30 Hz. Saved-frame counts and control ticks are different quantities.*

Open **Sensors**, enable the required outputs and enter position, rotation, resolution and field of view. Use live preview when needed; **Apply to preview** checks a draft, while **Save sensor settings** commits it for collection. Stop the sensor preview before collection.

![Actual sensor settings for the front camera and selected RGB, segmentation, depth and object annotations](images/walkthrough/sensor-settings.png)

*The verified run used front camera 400×300 and BEV 500×500. Numeric calibration, not the placement sketch alone, defines the recorded geometry.*

## 3. Collect and verify saved data

Return to collection, click **Start collection**, review the job snapshot and confirm. Watch actual saved-sample progress and saved RGB/BEV previews. When completed, check the final sample count, job state and logs. If a job fails, retain its outputs and inspect the reported engine/job error.

![Real running collection job with saved camera preview](images/walkthrough/collection-running.png)

![Completed real collection showing 80 out of 80 saved samples and BEV](images/walkthrough/collection-complete.png)

*This run finished 80/80 samples in about 24.49 seconds. No additional CARLA run was needed to produce this guide.*

## 4. Split scenes, choose a dataset and replay it

Open **Sequence analysis / Dataset**, assign complete scenes to TRAIN and VAL, give the new dataset a name, and select **Prepare dataset**. Preparation copies data and recorded calibration; it preserves the source. Use different scenes rather than mixing neighboring frames randomly. Check the resulting split counts.

![Actual scene selection preparing a dataset with separate TRAIN and VAL scenes](images/walkthrough/dataset-split.png)

*Here Town01 supplies TRAIN 40 and Town03 supplies VAL 40. Two different towns in this small demo do not establish broad coverage or generalization.*

Choose a sequence and camera, then press Play or move the Frame slider. Verify aligned RGB, segmentation, depth, object annotations and BEV for the selected Frame ID. Select another BEV layer only when supplied; heed the yellow asset/height warnings.

![Actual synchronized stored-frame playback with RGB, segmentation, depth, object boxes and BEV](images/walkthrough/sequence-playback.png)

*The shown VAL frame has zero object boxes. Metric BEV height is unavailable; a displayed ego or map-layer value does not create missing per-pixel road elevation.*

## 5. Select model, tasks, losses and training parameters

Open **Training → Setup** and select the prepared dataset from the TRAIN/VAL pair dropdown. Verify calibration and available inputs/targets. Select encoder and decoders, then enable only tasks supported by the data. Disabled tasks explain what is missing; readiness can still be incomplete at this stage.

![Actual model/task setup before the final training readiness check](images/walkthrough/model-tasks.png)

In **Task settings**, inspect observed segmentation classes and validity masks. Choose a supported loss for each enabled task and positive task weights; review the recommendation before applying class changes. Set epochs, batch size, learning rate, optimizer, input size and prediction-recording intervals. Save and run **Check readiness**; it verifies data, dependencies, CUDA and the selected official pretrained weights. A status read alone does not download weights. Start or enqueue only after readiness succeeds.

![Actual short demo configuration with 3 epochs, batch 4, Adam and 128 by 128 inputs](images/walkthrough/training-config.png)

*The demo used ResNet18 / FPN / BEV ResNet, Seg/Depth/OD/BEV, LR 0.0001 and 3 epochs. These are functional-demo settings, not recommended convergence settings. This capture precedes the final readiness action.*

## 6. Watch training and use checkpoints

Select **Monitoring** to inspect actual epoch/step progress, measured ETA, logs and saved metrics. Queueing retains immutable configuration/data/calibration snapshots. **Pause** finishes the current epoch; **Resume** continues from its checkpoint. **Start again** creates a new run from epoch 1. Stopped-job continuation requires a compatible completed checkpoint.

![Actual training monitor from the CUDA demo job](images/walkthrough/training-monitor.png)

*This job completed 3 epochs / global step 30 in about 30.21 seconds. The completed monitor verifies execution; it is not a model-quality benchmark.*

## 7. Browse predictions and interpret results

In Monitoring or **Results**, select an experiment, split, sample and task. Inspect **Input / Ground truth / Prediction** side by side. Move the Epoch or Iteration slider through saved positions only; missing positions are not synthesized. Validation samples are fixed per dataset content; comparison across experiments requires matching fingerprints.

![Real prediction browser showing validation input, segmentation ground truth and prediction](images/walkthrough/prediction-browser.png)

Scroll to loss graphs and task tables. Read the **TRAIN / VAL** legend and metric direction; Best records its epoch. Check saved configuration, `train_metrics.csv`, prediction history and best/last checkpoints in the experiment results. Queue deletion retains results.

![Actual TRAIN and VAL loss legends and per-task recorded metrics](images/walkthrough/metrics.png)

*Final VAL Seg mIoU 0.0341, depth RMSE 24.846 m, BEV mean IoU 0. VAL has no positive object boxes, so detection quality cannot be established from this split. The 252 prediction records verify saved-image exploration; this three-epoch run does not demonstrate convergence.*

For supported losses, restart behavior and recording controls see [training follow-up](TRAINING_UI_FOLLOWUP_KO.md). For precise BEV height limitations and Epic/Low behavior see [BEV integration](GLOBAL_MAP_INTEGRATION_KO.md) and [CARLA stability](CARLA_STABILITY_KO.md).

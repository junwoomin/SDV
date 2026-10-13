"""Prediction cadence: legacy snapshots keep their global-step behavior."""
def should_capture(monitor, epoch_step, global_step):
    interval = int(monitor.get('train_preview_every', 100))
    step = epoch_step if monitor.get('train_preview_basis', 'global') == 'epoch' else global_step
    return interval > 0 and step % interval == 0

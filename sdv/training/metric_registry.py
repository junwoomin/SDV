"""Metric labels and optimization directions, importable without PyTorch.

`min` means lower is better; `max` means higher is better. All ratio metrics
are stored as fractions, not percentages. Historical legacy binary-logit
segmentation IoU is deliberately a separate metric from multiclass mIoU.
"""


def _definition(name, task, direction, label, unit="ratio"):
    return {"name": name, "task": task, "direction": direction,
            "label": label, "unit": unit,
            "maximum": 1.0 if direction == "max" and unit == "ratio" else None,
            "detail": "_class_" in name}


METRIC_DEFINITIONS = {}
for _name, _task, _label in (
        ("loss", "All", "Total loss"),
        ("bev_loss", "BEV", "BEV loss"),
        ("das_loss", "BEV", "DAS loss"),
        ("lane_loss", "BEV", "Lane loss"),
        ("vehicle_loss", "BEV", "Vehicle loss"),
        ("walker_loss", "BEV", "Walker loss"),
        ("seg_loss", "Seg", "Segmentation loss"),
        ("depth_loss", "Depth", "Depth loss"),
        ("od_loss", "od", "Detection loss")):
    METRIC_DEFINITIONS[_name] = _definition(_name, _task, "min", _label, "loss")

for _name, _label in (("seg_miou", "mIoU"),
                      ("seg_pixel_accuracy", "Pixel Accuracy"),
                      ("seg_mean_dice", "mDice"),
                      ("bev_miou", "BEV mean IoU")):
    METRIC_DEFINITIONS[_name] = _definition(
        _name, "Seg" if _name.startswith("seg_") else "BEV", "max", _label)
for _class in range(6):
    _name = "seg_iou_class_{}".format(_class)
    METRIC_DEFINITIONS[_name] = _definition(
        _name, "Seg", "max", "Segmentation class {} IoU".format(_class))
for _task in ("das", "lane", "vehicle", "walker"):
    for _metric in ("iou", "dice", "f1", "precision", "recall"):
        _name = "bev_{}_{}".format(_task, _metric)
        METRIC_DEFINITIONS[_name] = _definition(
            _name, "BEV", "max", "BEV {} {}".format(_task.title(), _metric.upper()))
for _name, _label, _unit in (
        ("depth_abs_rel", "Depth AbsRel", "ratio"),
        ("depth_sq_rel", "Depth SqRel", "m"),
        ("depth_rmse", "Depth RMSE", "m"),
        ("depth_rmse_log", "Depth RMSE log", "log(m)"),
        ("depth_mae", "Depth MAE", "m")):
    METRIC_DEFINITIONS[_name] = _definition(_name, "Depth", "min", _label, _unit)
for _index in range(1, 4):
    _name = "depth_delta{}".format(_index)
    METRIC_DEFINITIONS[_name] = _definition(
        _name, "Depth", "max", "Depth delta{} (<1.25^{})".format(_index, _index))
for _name, _direction, _label, _unit in (
        ("depth_valid_pixels", "max", "Depth valid pixel count", "pixels"),
        ("depth_invalid_predictions", "min", "Depth invalid prediction count", "pixels"),
        ("depth_prediction_valid_ratio", "max", "Depth finite positive prediction ratio", "ratio")):
    METRIC_DEFINITIONS[_name] = _definition(_name, "Depth", _direction, _label, _unit)
for _name, _label in (
        ("od_ap50", "mAP@0.5"),
        ("od_ap50_95", "mAP@0.5:0.95"),
        ("od_precision", "Detection precision (IoU 0.50, score >=0.25)"),
        ("od_recall", "Detection recall (IoU 0.50, score >=0.25)"),
        ("od_f1", "Detection F1 (IoU 0.50, score >=0.25)")):
    METRIC_DEFINITIONS[_name] = _definition(_name, "od", "max", _label)
for _index, _label in enumerate(("Vehicle", "Pedestrian")):
    for _metric in ("ap50", "ap50_95"):
        _name = "od_{}_class_{}".format(_metric, _index)
        METRIC_DEFINITIONS[_name] = _definition(
            _name, "od", "max", "Detection {} {}".format(_label, _metric.upper()))


def _definitions(config=None):
    """Class labels must come from the immutable experiment configuration."""
    from sdv.data_contract import segmentation_spec, detection_spec
    settings = config.get('task_settings') if config else None
    definitions = {name: dict(value) for name, value in METRIC_DEFINITIONS.items()
                   if '_class_' not in name}
    for item in segmentation_spec(settings):
        for kind, label in (('iou', 'IoU'), ('dice', 'Dice')):
            name = 'seg_{}_class_{}'.format(kind, item['output_index'])
            definitions[name] = _definition(name, 'Seg', 'max', item['name'] + ' ' + label)
            definitions[name].update(class_id=item['id'], class_name=item['name'], color=item['color'])
    for item in detection_spec(settings):
        for kind, label in (('ap50', 'AP@0.5'), ('ap50_95', 'AP@0.5:0.95')):
            name = 'od_{}_class_{}'.format(kind, item['output_index'])
            definitions[name] = _definition(name, 'od', 'max', item['name'] + ' ' + label)
            definitions[name].update(class_id=item['id'], class_name=item['name'], color=item['color'])
    # Historical fields remain understandable, but the standard metrics above
    # are preferred. Their exact numeric history is not rewritten.
    for name, task, direction, label, unit in (
        ('abs_rel', 'Depth', 'min', 'Legacy AbsRel', 'ratio'),
        ('OD_iou', 'od', 'max', 'Legacy anchor IoU', 'ratio'),
        ('OD_loss', 'od', 'min', 'Detection loss', 'loss'),
        ('m_bev_iou', 'BEV', 'max', 'Legacy mean BEV IoU', 'ratio')):
        definitions[name] = _definition(name, task, direction, label, unit)
        definitions[name]['detail'] = True
        definitions[name]['legacy'] = True
    for name in ('das', 'lane', 'vehicle', 'walker'):
        key = name + '_iou'
        definitions[key] = _definition(key, 'BEV', 'max', 'Legacy ' + name + ' IoU')
        definitions[key]['detail'] = True
        definitions[key]['legacy'] = True
    return definitions


def metric_definition(name, config=None):
    base = name
    if base.startswith('train_'):
        base = base[6:]
    elif base.startswith('val_'):
        base = base[4:]
    definition = _definitions(config).get(base)
    if definition is None:
        # Direction is still known for historical dynamic class channels even
        # if the caller has not yet loaded that experiment's label schema.
        if base.startswith(('seg_iou_class_', 'seg_dice_class_', 'od_ap50_class_', 'od_ap50_95_class_')):
            return dict(_definition(name, 'Seg' if base.startswith('seg_') else 'od', 'max', base), detail=True)
        return None
    result = dict(definition)
    result['name'] = name
    return result


def metric_registry(config=None):
    definitions = _definitions(config)
    return {prefix + name: dict(value, name=prefix + name)
            for prefix in ('train_', 'val_') for name, value in definitions.items()}

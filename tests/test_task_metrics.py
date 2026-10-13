"""Analytical real-tensor checks; automatically skipped in the web-only env."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from sdv.training.metric_registry import metric_definition, metric_registry
from sdv.training.task_metrics import (
    BinaryMetrics, DepthMetrics, DetectionMetrics, EpochMetrics, SegmentationMetrics)


def _logits(indices, classes=6):
    index = torch.tensor(indices).reshape(1, 1, 1, -1)
    return torch.nn.functional.one_hot(index, classes).permute(0, 1, 4, 2, 3).float() * 8 - 4


def _onehot(indices, classes=6):
    index = torch.tensor(indices).reshape(1, 1, 1, -1)
    return torch.nn.functional.one_hot(index, classes).permute(0, 1, 4, 2, 3).float()


def test_segmentation_perfect_ignores_absent_classes():
    acc = SegmentationMetrics()
    acc.update(_logits([0, 1, 2, 2]), _onehot([0, 1, 2, 2]))
    metrics = acc.compute()
    assert metrics["seg_miou"] == 1
    assert metrics["seg_mean_dice"] == 1
    assert metrics["seg_pixel_accuracy"] == 1
    assert metrics["seg_iou_class_2"] == 1
    assert "seg_iou_class_5" not in metrics


def test_segmentation_multiclass_confusion_analytical():
    # Confusion [[1,1],[0,2]]: class IoU .5,2/3; Dice 2/3,.8.
    acc = SegmentationMetrics()
    acc.update(_logits([0, 1, 1, 1]), _onehot([0, 0, 1, 1]))
    metrics = acc.compute()
    assert metrics["seg_miou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics["seg_mean_dice"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["seg_pixel_accuracy"] == 0.75
    assert metrics["seg_iou_class_0"] == 0.5
    assert metrics["seg_iou_class_1"] == pytest.approx(2 / 3)


def test_segmentation_epoch_counts_are_pixel_weighted_not_batch_means():
    acc = SegmentationMetrics()
    acc.update(_logits([0]), _onehot([0]))
    acc.update(_logits([1, 1, 1]), _onehot([0, 0, 0]))
    assert acc.compute()["seg_pixel_accuracy"] == 0.25
    assert acc.compute()["seg_miou"] == 0.125


def test_segmentation_all_zero_gt_is_unlabelled():
    acc = SegmentationMetrics()
    acc.update(_logits([0, 1]), torch.zeros((1, 1, 6, 1, 2)))
    assert acc.compute() == {}


def test_segmentation_class_index_gt_and_ignored_index():
    acc = SegmentationMetrics()
    acc.update(_logits([0, 1, 2]), torch.tensor([[[[0, 1, -1]]]]))
    assert acc.compute()["seg_pixel_accuracy"] == 1
    assert "seg_iou_class_2" not in acc.compute()


def test_segmentation_shape_mismatch_is_error():
    acc = SegmentationMetrics()
    with pytest.raises(ValueError, match="one-hot or class"):
        acc.update(_logits([0, 1]), torch.zeros((1, 2)))


def test_binary_confusion_exact_and_logit_threshold_zero():
    acc = BinaryMetrics()
    acc.update(torch.tensor([1., 1., -1., -1.]), torch.tensor([1., 0., 1., 0.]))
    assert acc.compute() == {"iou": 1 / 3, "dice": 0.5, "f1": 0.5,
                             "precision": 0.5, "recall": 0.5}


def test_binary_empty_foreground_not_perfect():
    acc = BinaryMetrics()
    acc.update(torch.tensor([-1., 0.]), torch.tensor([0., 0.]))
    assert acc.compute() == {}


def test_binary_missed_foreground_returns_zero_overlap_no_fake_precision():
    acc = BinaryMetrics()
    acc.update(torch.tensor([-1., -1.]), torch.tensor([1., 0.]))
    assert acc.compute() == {"iou": 0, "dice": 0, "f1": 0, "recall": 0}


def test_bev_class_means_exclude_empty_foreground():
    acc = EpochMetrics({"BEV": True})
    acc.update({"DAS": torch.tensor([1.]), "Lane": torch.tensor([-1.])},
               {"das_GT": torch.tensor([1.]), "lane_GT": torch.tensor([0.])})
    assert acc.compute("val")["val_bev_miou"] == 1
    assert "val_bev_lane_iou" not in acc.compute("val")


def test_depth_perfect_metrics_and_delta():
    acc = DepthMetrics()
    acc.update(torch.tensor([1., 5., 10.]), torch.tensor([1., 5., 10.]))
    metrics = acc.compute()
    for name in ("abs_rel", "sq_rel", "rmse", "rmse_log", "mae"):
        assert metrics["depth_" + name] == 0
    for index in (1, 2, 3):
        assert metrics["depth_delta{}".format(index)] == 1
    assert metrics["depth_prediction_valid_ratio"] == 1


def test_depth_analytical_values_and_pixel_weighted_sum():
    acc = DepthMetrics()
    acc.update(torch.tensor([2.]), torch.tensor([1.]))
    acc.update(torch.tensor([2., 3.]), torch.tensor([2., 2.]))
    metrics = acc.compute()
    assert metrics["depth_abs_rel"] == 0.5
    assert metrics["depth_sq_rel"] == 0.5
    assert metrics["depth_rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert metrics["depth_mae"] == pytest.approx(2 / 3)
    assert metrics["depth_rmse_log"] == pytest.approx(
        math.sqrt((math.log(2) ** 2 + math.log(1.5) ** 2) / 3))
    assert metrics["depth_delta1"] == pytest.approx(1 / 3)
    assert metrics["depth_delta2"] == pytest.approx(2 / 3)
    assert metrics["depth_delta3"] == pytest.approx(2 / 3)


def test_depth_mask_range_nonfinite_and_invalid_predictions():
    acc = DepthMetrics()
    acc.update(torch.tensor([1., 20., 5., float("nan"), 1., 2., 0.]),
               torch.tensor([1., 2., 0., 4., float("nan"), 101., 2.]),
               torch.tensor([True, False, True, True, True, True, True]))
    metrics = acc.compute()
    assert metrics["depth_valid_pixels"] == 1
    assert metrics["depth_invalid_predictions"] == 2
    assert metrics["depth_prediction_valid_ratio"] == pytest.approx(1 / 3)
    assert metrics["depth_rmse"] == 0


def test_depth_zero_valid_pixels_never_nan_or_perfect():
    acc = DepthMetrics()
    acc.update(torch.tensor([1.]), torch.tensor([0.]))
    assert acc.compute() == {}
    acc.update(torch.tensor([float("nan")]), torch.tensor([1.]))
    assert acc.compute() == {"depth_valid_pixels": 0., "depth_invalid_predictions": 1.,
                             "depth_prediction_valid_ratio": 0.}


def _image(acc, boxes, scores, labels, gt_boxes, gt_labels):
    acc.update_image(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                     torch.tensor(scores), torch.tensor(labels),
                     torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
                     torch.tensor(gt_labels))


def test_detection_perfect_two_classes_all_thresholds():
    acc = DetectionMetrics()
    _image(acc, [[0, 0, 10, 10], [20, 20, 25, 25]], [0.9, 0.8], [0, 1],
           [[0, 0, 10, 10], [20, 20, 25, 25]], [0, 1])
    result = acc.compute()
    for name in ("od_ap50", "od_ap50_95", "od_precision", "od_recall", "od_f1"):
        assert result[name] == 1


def test_detection_false_positive_before_true_analytical_ap():
    acc = DetectionMetrics()
    _image(acc, [[20, 20, 30, 30], [0, 0, 10, 10]], [0.95, 0.9], [0, 0],
           [[0, 0, 10, 10]], [0])
    result = acc.compute()
    assert result["od_ap50"] == 0.5
    assert result["od_ap50_95"] == 0.5
    assert result["od_precision"] == 0.5
    assert result["od_recall"] == 1
    assert result["od_f1"] == pytest.approx(2 / 3)


def test_detection_iou_thresholds_have_actual_difference():
    acc = DetectionMetrics()
    # 60/100=.6 intersection/union: succeeds at .50,.55,.60 only.
    _image(acc, [[0, 0, 6, 10]], [0.9], [0], [[0, 0, 10, 10]], [0])
    result = acc.compute()
    assert result["od_ap50"] == 1
    assert result["od_ap50_95"] == pytest.approx(0.3)


def test_detection_nms_is_class_aware():
    acc = DetectionMetrics()
    # Identical boxes from different classes are both kept, duplicate same
    # class is suppressed.
    _image(acc, [[0, 0, 10, 10]] * 3, [0.95, 0.9, 0.8], [0, 1, 0],
           [[0, 0, 10, 10]] * 2, [0, 1])
    assert acc.compute()["od_precision"] == 1
    assert acc.compute()["od_ap50"] == 1


def test_detection_empty_and_no_prediction_split():
    acc = DetectionMetrics()
    _image(acc, [], [], [], [], [])
    assert acc.compute() == {}
    _image(acc, [], [], [], [[0, 0, 1, 1]], [0])
    result = acc.compute()
    assert result["od_ap50"] == 0
    assert result["od_recall"] == 0
    assert result["od_f1"] == 0
    assert "od_precision" not in result
    assert "od_ap50_class_1" not in result


def test_detection_wrong_class_is_not_true_positive():
    acc = DetectionMetrics()
    _image(acc, [[0, 0, 10, 10]], [0.99], [1], [[0, 0, 10, 10]], [0])
    result = acc.compute()
    assert result["od_ap50"] == 0
    assert result["od_recall"] == 0
    assert result["od_precision"] == 0


def test_detection_low_score_affects_ap_but_not_precision_operating_point():
    acc = DetectionMetrics()
    _image(acc, [[0, 0, 10, 10]], [0.1], [0], [[0, 0, 10, 10]], [0])
    result = acc.compute()
    assert result["od_ap50"] == 1
    assert result["od_recall"] == 0
    assert "od_precision" not in result


def test_detection_box_decode_and_batch_camera_flattening():
    acc = DetectionMetrics()
    # Existing anchors are y1,x1,y2,x2; decoding emits x1,y1,x2,y2.
    classification = torch.tensor([[[0.9, 0.01]], [[0.01, 0.8]]])
    regression = torch.zeros((2, 1, 4))
    anchors = torch.tensor([[[2., 4., 12., 14.]]])
    annotations = torch.tensor([[[[4., 2., 14., 12., 0.]], [[4., 2., 14., 12., 1.]]]])
    images = torch.zeros((1, 2, 3, 32, 32))
    acc.update(classification, regression, anchors, annotations, images)
    assert acc.compute()["od_ap50"] == 1
    assert acc.images == 2


def test_detection_maximum_per_image_limits_state():
    acc = DetectionMetrics(max_detections=2)
    _image(acc, [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
           [0.9, 0.8, 0.7], [0, 0, 0], [[0, 0, 1, 1]], [0])
    assert sum(len(scores) for scores, _ in acc.rows[0]) == 2
    assert acc.compute()["od_precision"] == 0.5


def test_epoch_tuple_api_and_prefix_does_not_modify_gradients():
    depth = torch.tensor([[[[[2.]]]]], requires_grad=True)
    seg = _logits([1]).requires_grad_()
    outputs = (None, None, None, None, seg, depth, None, None)
    batch = [None] * 18
    batch[14], batch[15], batch[16] = _onehot([1]), torch.tensor([[[[[2.]]]]]), torch.ones_like(depth).bool()
    acc = EpochMetrics({"Seg": True, "Depth": True})
    acc.update(outputs, tuple(batch))
    values = acc.compute(prefix="val")
    assert values["val_seg_miou"] == 1
    assert values["val_depth_abs_rel"] == 0
    assert depth.grad is None
    assert seg.grad is None
    assert depth.requires_grad and seg.requires_grad
    assert all(math.isfinite(value) for value in values.values())


def test_disabled_tasks_are_absent_not_zero():
    acc = EpochMetrics({"Seg": True})
    acc.update({"seg": _logits([0])}, {"seg_GT": _onehot([0])})
    values = acc.compute("train_")
    assert values["train_seg_miou"] == 1
    assert not any(key.startswith("train_depth") or key.startswith("train_bev") for key in values)


def test_explicit_empty_tasks_and_wrong_tuple_are_not_silently_inferred():
    acc = EpochMetrics({})
    acc.update({"seg": _logits([0])}, {"seg_GT": _onehot([0])})
    assert acc.compute() == {}
    with pytest.raises(ValueError, match="8-output"):
        acc.update((None,), tuple([None] * 18))
    with pytest.raises(ValueError, match="18-value"):
        acc.update(tuple([None] * 8), (None,))


def test_detection_missing_anchors_is_clear_error():
    acc = EpochMetrics({"od": True})
    with pytest.raises(ValueError, match="anchors"):
        acc.update({"classification": torch.zeros((1, 1, 2))}, {})


def test_registry_defines_every_emitted_common_metric_and_best_direction():
    definition = metric_definition("val_depth_rmse")
    assert definition["direction"] == "min"
    assert definition["unit"] == "m"
    assert metric_definition("train_seg_miou")["direction"] == "max"
    assert metric_definition("val_od_ap50_95")["direction"] == "max"
    assert metric_definition("val_seg_iou") is None  # legacy incompatible metric
    registry = metric_registry()
    assert "val_bev_das_iou" in registry
    assert "train_loss" in registry
    assert "val_od_ap50_class_1" in registry

"""Actual epoch-level perception metrics with bounded per-image detection state.

Segmentation: multiclass argmax and a global confusion matrix; absent classes
with zero union are omitted from the class mean (never assigned perfect IoU).
BEV: four independent binary masks, logits > 0, global pixel counts.
Depth: pixel-weighted sums on the supplied validity mask and finite ground
truth in (0, 100] m. Invalid/nonpositive predictions are counted explicitly;
error metrics use finite positive pairs and are absent when no pair is valid.
Detection: class-aware NMS at 0.5, AP score cutoff 0.001, max 100 detections
per image, 101-point precision interpolation at IoU 0.50:0.05:0.95. The
dataset has no crowd flags/area categories, so these are SDV all-area AP
metrics, not the complete COCO evaluation protocol. P/R/F1 use score >=0.25.

Definitions checked against primary implementations:
https://docs.pytorch.org/ignite/generated/ignite.metrics.IoU.html
https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/metric.py
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
"""
import math

import numpy as np
import torch


_BATCH_NAMES = (
    "imgs", "imgs_ori", "das_GT", "lane_GT", "vehicle_GT", "walker_GT",
    "lidar_data", "lidar_mask", "post_rots", "post_trans", "extrinsic",
    "intrinsic", "rotation", "translation", "seg_GT", "dept_GT",
    "depth_mask", "annotations")
_OUTPUT_NAMES = ("DAS", "Lane", "Vehicle", "walker", "seg", "depth",
                 "classification", "regression")


def _tensor(value, device=None):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device) if device is not None else value.detach()
    return torch.as_tensor(value, device=device)


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _insert(result, name, value):
    if value is not None and math.isfinite(float(value)):
        result[name] = float(value)


class BinaryMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits, target):
        pred = _tensor(logits)
        truth = _tensor(target, pred.device)
        if pred.shape != truth.shape:
            raise ValueError("BEV prediction and target shapes differ")
        valid = torch.isfinite(pred) & torch.isfinite(truth)
        pred, truth = pred[valid] > 0, truth[valid] > 0.5
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())
        self.tn += int((~pred & ~truth).sum().item())

    def compute(self):
        result = {}
        _insert(result, "iou", _ratio(self.tp, self.tp + self.fp + self.fn))
        dice = _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        _insert(result, "dice", dice)
        _insert(result, "f1", dice)
        _insert(result, "precision", _ratio(self.tp, self.tp + self.fp))
        _insert(result, "recall", _ratio(self.tp, self.tp + self.fn))
        return result


class SegmentationMetrics:
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits, target):
        scores = _tensor(logits)
        truth = _tensor(target, scores.device)
        # Scores may be B,N,C,H,W or B,C,H,W; class is always dimension -3.
        if scores.ndim not in (4, 5) or scores.shape[-3] != self.num_classes:
            raise ValueError("Segmentation expects B[,N],C,H,W logits")
        pred = scores.argmax(dim=-3)
        finite = torch.isfinite(scores).all(dim=-3)
        if truth.shape == scores.shape:
            # An all-zero label vector is unlabelled, not class zero.
            valid = torch.isfinite(truth).all(dim=-3) & (truth.sum(dim=-3) > 0)
            truth = truth.argmax(dim=-3)
        elif truth.shape == pred.shape:
            valid = torch.isfinite(truth) & (truth >= 0) & (truth < self.num_classes)
            truth = truth.to(torch.int64)
        else:
            raise ValueError("Segmentation target must be one-hot or class indices")
        valid &= finite
        index = truth[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(index, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self):
        matrix = self.confusion.numpy().astype(np.float64)
        tp = np.diag(matrix)
        actual, predicted = matrix.sum(1), matrix.sum(0)
        union = actual + predicted - tp
        result = {}
        present = union > 0
        if np.any(present):
            result["seg_miou"] = float(np.mean(tp[present] / union[present]))
            result["seg_mean_dice"] = float(np.mean(
                2 * tp[present] / (actual[present] + predicted[present])))
        _insert(result, "seg_pixel_accuracy", _ratio(tp.sum(), matrix.sum()))
        for index in range(self.num_classes):
            _insert(result, "seg_iou_class_{}".format(index), _ratio(tp[index], union[index]))
            _insert(result, "seg_dice_class_{}".format(index), _ratio(2 * tp[index], actual[index] + predicted[index]))
        return result


class DepthMetrics:
    def __init__(self, max_depth=100.0):
        self.max_depth = max_depth
        self.count = self.gt_count = self.invalid_predictions = 0
        self.abs_rel = self.sq_rel = self.squared = self.log_squared = self.absolute = 0.0
        self.delta = [0, 0, 0]

    @torch.no_grad()
    def update(self, prediction, target, mask=None):
        pred = _tensor(prediction)
        truth = _tensor(target, pred.device)
        if pred.shape != truth.shape:
            raise ValueError("Depth prediction and target shapes differ")
        valid_gt = torch.isfinite(truth) & (truth > 0) & (truth <= self.max_depth)
        if mask is not None:
            requested = _tensor(mask, pred.device).bool()
            if requested.shape != truth.shape:
                raise ValueError("Depth validity mask shape differs from target")
            valid_gt &= requested
        valid_pred = torch.isfinite(pred) & (pred > 0)
        self.gt_count += int(valid_gt.sum().item())
        self.invalid_predictions += int((valid_gt & ~valid_pred).sum().item())
        valid = valid_gt & valid_pred
        count = int(valid.sum().item())
        if not count:
            return
        pred, truth = pred[valid].double(), truth[valid].double()
        difference = pred - truth
        squared = difference.square()
        self.count += count
        self.absolute += float(difference.abs().sum().item())
        self.squared += float(squared.sum().item())
        self.abs_rel += float((difference.abs() / truth).sum().item())
        self.sq_rel += float((squared / truth).sum().item())
        self.log_squared += float((pred.log() - truth.log()).square().sum().item())
        ratio = torch.maximum(pred / truth, truth / pred)
        for index in range(3):
            self.delta[index] += int((ratio < 1.25 ** (index + 1)).sum().item())

    def compute(self):
        result = {}
        if self.gt_count:
            result["depth_valid_pixels"] = float(self.count)
            result["depth_invalid_predictions"] = float(self.invalid_predictions)
            result["depth_prediction_valid_ratio"] = float(self.count) / self.gt_count
        if not self.count:
            return result
        result.update({
            "depth_abs_rel": self.abs_rel / self.count,
            "depth_sq_rel": self.sq_rel / self.count,
            "depth_rmse": math.sqrt(self.squared / self.count),
            "depth_rmse_log": math.sqrt(self.log_squared / self.count),
            "depth_mae": self.absolute / self.count})
        for index in range(3):
            result["depth_delta{}".format(index + 1)] = float(self.delta[index]) / self.count
        return result


def _box_iou(boxes, gt):
    """Numpy xyxy IoU, no inclusive +1 pixel convention."""
    intersection_min = np.maximum(boxes[:, None, :2], gt[None, :, :2])
    intersection_max = np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    intersection = np.prod(np.maximum(0, intersection_max - intersection_min), axis=2)
    area_a = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0, gt[:, 2:] - gt[:, :2]), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class DetectionMetrics:
    def __init__(self, num_classes=2, score_threshold=0.001,
                 precision_threshold=0.25, nms_threshold=0.5, max_detections=100):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.precision_threshold = precision_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.iou_thresholds = np.arange(50, 100, 5, dtype=np.float64) / 100.0
        self.gt_counts = np.zeros(num_classes, dtype=np.int64)
        # Only post-NMS scores and 10 TP flags survive, not model/GPU tensors.
        self.rows = [[] for _ in range(num_classes)]
        self.images = 0

    @torch.no_grad()
    def update_image(self, boxes, scores, labels, gt_boxes, gt_labels):
        from torchvision.ops import nms
        boxes = _tensor(boxes).cpu().float().reshape(-1, 4)
        scores = _tensor(scores).cpu().float().reshape(-1)
        labels = _tensor(labels).cpu().long().reshape(-1)
        gt_boxes = _tensor(gt_boxes).cpu().float().reshape(-1, 4)
        gt_labels = _tensor(gt_labels).cpu().long().reshape(-1)
        if boxes.shape[0] != scores.numel() or scores.numel() != labels.numel():
            raise ValueError("Detection predictions have different lengths")
        if gt_boxes.shape[0] != gt_labels.numel():
            raise ValueError("Detection annotations have different lengths")
        valid_gt = torch.isfinite(gt_boxes).all(1) & (gt_boxes[:, 2] > gt_boxes[:, 0]) & (
            gt_boxes[:, 3] > gt_boxes[:, 1]) & (gt_labels >= 0) & (gt_labels < self.num_classes)
        gt_boxes, gt_labels = gt_boxes[valid_gt], gt_labels[valid_gt]
        valid_pred = torch.isfinite(boxes).all(1) & torch.isfinite(scores) & (
            boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (
            labels >= 0) & (labels < self.num_classes) & (scores >= self.score_threshold)
        boxes, scores, labels = boxes[valid_pred], scores[valid_pred], labels[valid_pred]
        keeps = []
        for cls in range(self.num_classes):
            candidates = torch.nonzero(labels == cls, as_tuple=False).flatten()
            # Model anchor tensors can be large; prefilter to highest 1,000
            # candidates/class before CPU NMS, independent of dataset length.
            if candidates.numel() > 1000:
                candidates = candidates[scores[candidates].topk(1000).indices]
            if candidates.numel():
                keeps.append(candidates[nms(boxes[candidates], scores[candidates], self.nms_threshold)])
        if keeps:
            kept = torch.cat(keeps)
            kept = kept[scores[kept].argsort(descending=True)[:self.max_detections]]
            boxes, scores, labels = boxes[kept], scores[kept], labels[kept]
        else:
            boxes, scores, labels = boxes[:0], scores[:0], labels[:0]
        self.images += 1
        for cls in range(self.num_classes):
            selected = labels == cls
            truth = gt_boxes[gt_labels == cls].numpy()
            self.gt_counts[cls] += truth.shape[0]
            predictions = boxes[selected].numpy()
            cls_scores = scores[selected].numpy()
            correct = np.zeros((len(predictions), len(self.iou_thresholds)), dtype=np.bool_)
            if len(predictions) and len(truth):
                overlaps = _box_iou(predictions, truth)
                for threshold_index, threshold in enumerate(self.iou_thresholds):
                    matched = np.zeros(len(truth), dtype=np.bool_)
                    # Predictions are sorted by descending confidence; match
                    # each to its best yet-unmatched GT at this IoU threshold.
                    for pred_index in range(len(predictions)):
                        available = np.where(~matched, overlaps[pred_index], -1.0)
                        target_index = int(available.argmax())
                        if available[target_index] >= threshold:
                            correct[pred_index, threshold_index] = True
                            matched[target_index] = True
            if len(predictions):
                self.rows[cls].append((cls_scores.copy(), correct))

    @torch.no_grad()
    def update(self, classifications, regressions, anchors, annotations, images):
        from sdv.training.modeling.utils import BBoxTransform, ClipBoxes
        classification = _tensor(classifications)
        regression = _tensor(regressions, classification.device)
        images = _tensor(images, classification.device)
        if images.ndim == 5:
            images = images.flatten(0, 1)
        truth = _tensor(annotations)
        if truth.ndim == 4:
            truth = truth.flatten(0, 1)
        if classification.ndim != 3 or classification.shape[-1] != self.num_classes:
            raise ValueError("Detection classification expects I,A,C probabilities")
        if images.shape[0] != classification.shape[0] or truth.shape[0] != images.shape[0]:
            raise ValueError("Detection camera/image/annotation batch counts differ")
        anchor_tensor = _tensor(anchors, classification.device)
        boxes = BBoxTransform()(anchor_tensor, regression)
        boxes = ClipBoxes()(boxes, images)
        for index in range(classification.shape[0]):
            # Multi-label class scores are evaluated independently; selecting
            # only argmax would silently discard true secondary class scores.
            selected = classification[index] >= self.score_threshold
            anchor_index, class_index = selected.nonzero(as_tuple=True)
            gt = truth[index]
            valid = gt[:, 4] >= 0
            self.update_image(boxes[index, anchor_index], classification[index][selected],
                              class_index, gt[valid, :4], gt[valid, 4])

    def compute(self):
        result, class_aps = {}, []
        true_positive = false_positive = false_negative = 0
        for cls in range(self.num_classes):
            if self.rows[cls]:
                scores = np.concatenate([row[0] for row in self.rows[cls]])
                correct = np.concatenate([row[1] for row in self.rows[cls]])
                order = np.argsort(-scores, kind="mergesort")
                scores, correct = scores[order], correct[order]
            else:
                scores = np.zeros(0, dtype=np.float32)
                correct = np.zeros((0, len(self.iou_thresholds)), dtype=np.bool_)
            relevant = scores >= self.precision_threshold
            tp = int(correct[relevant, 0].sum())
            true_positive += tp
            false_positive += int(relevant.sum()) - tp
            false_negative += int(self.gt_counts[cls]) - tp
            if not self.gt_counts[cls]:
                continue  # No GT for a class: AP is undefined, not perfect.
            if len(scores):
                tps = np.cumsum(correct, axis=0, dtype=np.float64)
                fps = np.cumsum(~correct, axis=0, dtype=np.float64)
                recall = tps / float(self.gt_counts[cls])
                precision = tps / (tps + fps)
                precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
                aps = []
                for threshold_index in range(len(self.iou_thresholds)):
                    indices = np.searchsorted(recall[:, threshold_index], np.linspace(0, 1, 101), side="left")
                    interpolated = np.zeros(101)
                    available = indices < len(precision)
                    interpolated[available] = precision[indices[available], threshold_index]
                    aps.append(float(interpolated.mean()))
                ap = np.asarray(aps)
            else:
                ap = np.zeros(len(self.iou_thresholds))
            result["od_ap50_class_{}".format(cls)] = float(ap[0])
            result["od_ap50_95_class_{}".format(cls)] = float(ap.mean())
            class_aps.append(ap)
        if class_aps:
            mean = np.mean(np.stack(class_aps), axis=0)
            result["od_ap50"] = float(mean[0])
            result["od_ap50_95"] = float(mean.mean())
        # On a wholly empty GT split the recall/AP/F1 are undefined; precision
        # still reports zero when detections are false positives.
        _insert(result, "od_precision", _ratio(true_positive, true_positive + false_positive))
        _insert(result, "od_recall", _ratio(true_positive, true_positive + false_negative))
        if int(self.gt_counts.sum()):
            _insert(result, "od_f1", _ratio(2 * true_positive,
                    2 * true_positive + false_positive + false_negative))
        return result


class EpochMetrics:
    """One independent accumulator per training/validation epoch.

    update accepts the unchanged model's 8-output tuple and unchanged
    collater's 18-value tuple, or dictionaries with the same names. No model
    parameters, RNG, gradients or training loss are modified.
    """
    def __init__(self, model_conf=None, task_settings=None):
        self.tasks = (dict(model_conf) if model_conf is not None else
                      {"BEV": True, "Seg": True, "Depth": True, "od": True})
        from sdv.data_contract import normalize_task_settings, segmentation_spec
        settings = normalize_task_settings(task_settings)
        self.bev = {name: BinaryMetrics() for name in settings['bev']['layers']}
        self.segmentation = SegmentationMetrics(len(segmentation_spec(settings)))
        self.depth = DepthMetrics(max_depth=settings['depth']['max_m'])
        self.detection = DetectionMetrics(num_classes=len(settings['detection']['class_ids']))

    @torch.no_grad()
    def update(self, outputs, batch, anchors=None):
        if not isinstance(outputs, dict) and len(outputs) != len(_OUTPUT_NAMES):
            raise ValueError("Perception model must provide its original 8-output tuple")
        if not isinstance(batch, dict) and len(batch) != len(_BATCH_NAMES):
            raise ValueError("Metrics require the original 18-value collated batch")
        outputs = outputs if isinstance(outputs, dict) else dict(zip(_OUTPUT_NAMES, outputs))
        batch = batch if isinstance(batch, dict) else dict(zip(_BATCH_NAMES, batch))
        if self.tasks.get("BEV"):
            for output, target, name in (
                    ("DAS", "das_GT", "das"), ("Lane", "lane_GT", "lane"),
                    ("Vehicle", "vehicle_GT", "vehicle"), ("walker", "walker_GT", "walker")):
                if name in self.bev and outputs.get(output) is not None and batch.get(target) is not None:
                    self.bev[name].update(outputs[output], batch[target])
        if self.tasks.get("Seg") and outputs.get("seg") is not None and batch.get("seg_GT") is not None:
            self.segmentation.update(outputs["seg"], batch["seg_GT"])
        if self.tasks.get("Depth") and outputs.get("depth") is not None and batch.get("dept_GT") is not None:
            self.depth.update(outputs["depth"], batch["dept_GT"], batch.get("depth_mask"))
        if self.tasks.get("od") and outputs.get("classification") is not None:
            if anchors is None or batch.get("annotations") is None or batch.get("imgs") is None:
                raise ValueError("Detection metrics require anchors, annotations, and original images")
            self.detection.update(outputs["classification"], outputs["regression"], anchors,
                                  batch["annotations"], batch["imgs"])

    def compute(self, prefix=None):
        result, ious = {}, []
        if self.tasks.get("BEV"):
            for name, accumulator in self.bev.items():
                values = accumulator.compute()
                result.update({"bev_{}_{}".format(name, key): value for key, value in values.items()})
                if "iou" in values:
                    ious.append(values["iou"])
            if ious:
                result["bev_miou"] = float(sum(ious)) / len(ious)
        if self.tasks.get("Seg"):
            result.update(self.segmentation.compute())
        if self.tasks.get("Depth"):
            result.update(self.depth.compute())
        if self.tasks.get("od"):
            result.update(self.detection.compute())
        if prefix:
            prefix = prefix if prefix.endswith("_") else prefix + "_"
            return {prefix + key: value for key, value in result.items()}
        return result

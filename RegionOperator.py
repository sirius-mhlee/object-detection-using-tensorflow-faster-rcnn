import numpy as np

import Configuration as cfg

import AnchorOperator as ao
import BBoxOperator as bo

def nms_region(rpn_cls_prob, rpn_bbox_pred, trainable):
    anchors = ao.generate_anchors()

    rpn_cls_prob = np.transpose(rpn_cls_prob, [0, 3, 1, 2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2])

    cls_probs = rpn_cls_prob[:, cfg.anchor_num:, :, :]
    bbox_preds = rpn_bbox_pred

    prob_height, prob_width = cls_probs.shape[-2:]

    anchor_shift_x = np.arange(0, prob_width) * cfg.anchor_shift_value
    anchor_shift_y = np.arange(0, prob_height) * cfg.anchor_shift_value
    anchor_shift_x, anchor_shift_y = np.meshgrid(anchor_shift_x, anchor_shift_y)
    anchor_shifts = np.vstack((anchor_shift_x.ravel(), anchor_shift_y.ravel(), anchor_shift_x.ravel(), anchor_shift_y.ravel())).transpose()
    anchor_shift_num = anchor_shifts.shape[0]

    anchors = anchors.reshape((1, cfg.anchor_num, 4)) + anchor_shifts.reshape((1, anchor_shift_num, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((cfg.anchor_num * anchor_shift_num, 4))

    cls_probs = cls_probs.transpose((0, 2, 3, 1)).reshape((-1, 1))
    bbox_preds = bbox_preds.transpose((0, 2, 3, 1)).reshape((-1, 4))

    proposals = bo.transform_bbox_detect(anchors, bbox_preds)
    proposals = bo.clip_bbox(proposals)

    if trainable:
        pre_nms_topN = cfg.anchor_pre_nms_topN_train
        post_nms_topN = cfg.anchor_post_nms_topN_train
        nms_thresh = cfg.anchor_nms_thresh_train
        min_size = cfg.anchor_min_size_train
    else:
        pre_nms_topN = cfg.anchor_pre_nms_topN_detect
        post_nms_topN = cfg.anchor_post_nms_topN_detect
        nms_thresh = cfg.anchor_nms_thresh_detect
        min_size = cfg.anchor_min_size_detect

    keep = bo.filter_bbox(proposals, min_size)
    proposals = proposals[keep, :]
    cls_probs = cls_probs[keep]

    order = cls_probs.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    cls_probs = cls_probs[order]
    
    keep = bo.nms_bbox(np.hstack((proposals, cls_probs)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    cls_probs = cls_probs[keep]

    batch_ids = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    nms_region = np.hstack((batch_ids, proposals.astype(np.float32, copy=False)))
    
    return nms_region

def regression_target_region(target_rois):
    class_list = target_rois[:, 0]
    region_targets = np.zeros((class_list.size, cfg.object_class_num_with_bg * 4), dtype=np.float32)
    region_inside_weights = np.zeros(region_targets.shape, dtype=np.float32)
    
    idx = np.where(class_list < cfg.object_class_num)[0]
    for i in idx:
        class_idx = class_list[i]
        
        start = int(4 * class_idx)
        end = start + 4

        region_targets[i, start:end] = target_rois[i, 1:]
        region_inside_weights[i, start:end] = (1, 1, 1, 1)

    return region_targets, region_inside_weights

def target_region(nms_region, region_holder):
    zeros = np.zeros((region_holder.shape[0], 1), dtype=region_holder.dtype)
    region = np.vstack((nms_region, np.hstack((zeros, region_holder[:, :-1]))))

    overlaps = bo.overlap_bbox(np.ascontiguousarray(region[:, 1:5], dtype=np.float), np.ascontiguousarray(region_holder[:, :4], dtype=np.float))

    max_overlaps = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels = region_holder[gt_assignment, 4]

    fg_idx = np.where(max_overlaps >= cfg.region_train_fg_thresh)[0]
    fg_rois_per_image = np.round(cfg.region_train_fg_ratio * cfg.region_train_batch_size).astype(np.int32)
    fg_rois_per_image = min(fg_rois_per_image, fg_idx.size)
    if fg_idx.size > 0:
        fg_idx = np.random.choice(fg_idx, size=fg_rois_per_image, replace=False)

    bg_idx = np.where((max_overlaps >= cfg.region_train_bg_thresh_lo) & (max_overlaps < cfg.region_train_bg_thresh_hi))[0]
    bg_rois_per_image = cfg.region_train_batch_size - fg_rois_per_image
    bg_rois_per_image = min(bg_rois_per_image, bg_idx.size)
    if bg_idx.size > 0:
        bg_idx = np.random.choice(bg_idx, size=bg_rois_per_image, replace=False)

    keep_idx = np.append(fg_idx, bg_idx)

    labels = labels[keep_idx]
    labels[fg_rois_per_image:] = cfg.object_class_num
    
    rois = region[keep_idx]
    transform_rois = bo.transform_bbox_train(rois[:, 1:5], region_holder[gt_assignment[keep_idx], :4])
    target_rois = np.hstack((labels[:, np.newaxis], transform_rois)).astype(np.float32, copy=False)

    region_targets, region_inside_weights = regression_target_region(target_rois)

    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1, 1)
    region_targets = region_targets.reshape(-1, cfg.object_class_num_with_bg * 4)
    region_inside_weights = region_inside_weights.reshape(-1, cfg.object_class_num_with_bg * 4)
    region_outside_weights = np.array(region_inside_weights > 0).astype(np.float32)

    return rois, labels, region_targets, region_inside_weights, region_outside_weights
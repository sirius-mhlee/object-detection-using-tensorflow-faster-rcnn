import numpy as np

import Configuration as cfg

import AnchorOperator as ao
import BBoxOperator as bo

def nms_region(rpn_cls_prob, rpn_bbox_pred):
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
        pre_nms_topN = cfg.anchor_pre_nms_topN_detect
        post_nms_topN = cfg.anchor_post_nms_topN_detect
        nms_thresh = cfg.anchor_nms_thresh_detect
        min_size = cfg.anchor_min_size_detect
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

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1, 1, 1, 1)
    return bbox_targets, bbox_inside_weights

def target_region(nms_region, region_holder):
    zeros = np.zeros((region_holder.shape[0], 1), dtype=region_holder.dtype)
    all_rois = np.vstack((nms_region, np.hstack((zeros, region_holder[:, :-1]))))

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE // num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int32)

    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    targets = bbox_transform(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4])
    bbox_target_data = np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    rois = rois.reshape(-1,5)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
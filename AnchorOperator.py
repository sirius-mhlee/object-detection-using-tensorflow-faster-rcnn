import numpy as np

import Configuration as cfg

import BBoxOperator as bo

def get_anchor_rect_info(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    cx = anchor[0] + 0.5 * (w - 1)
    cy = anchor[1] + 0.5 * (h - 1)
    return w, h, cx, cy

def make_anchor_rect(w, h, cx, cy):
    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    anchor = np.hstack((cx - 0.5 * (w - 1), cy - 0.5 * (h - 1), cx + 0.5 * (w - 1), cy + 0.5 * (h - 1)))
    return anchor

def ratio_enum(anchor, ratios):
    w, h, cx, cy = get_anchor_rect_info(anchor)
    size = w * h
    size_ratios = size / ratios
    nw = np.round(np.sqrt(size_ratios))
    nh = np.round(nw * ratios)
    anchor = make_anchor_rect(nw, nh, cx, cy)
    return anchor

def scale_enum(anchor, scales):
    w, h, cx, cy = get_anchor_rect_info(anchor)
    nw = w * scales
    nh = h * scales
    anchor = make_anchor_rect(nw, nh, cx, cy)
    return anchor

def generate_anchors():
    base_anchor = np.array([1, 1, cfg.anchor_base_size, cfg.anchor_base_size]) - 1
    ratio_anchors = ratio_enum(base_anchor, cfg.anchor_ratios)
    anchors = np.vstack([scale_enum(ratio_anchors[i], np.array(cfg.anchor_scales)) for i in range(ratio_anchors.shape[0])])
    return anchors

def unmap_anchors(data, count, inds, fill):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def target_anchors(bbox_holder, rpn_cls_score):
    anchors = generate_anchors()
    
    score_height, score_width = rpn_cls_score.shape[1:3]

    anchor_shift_x = np.arange(0, score_width) * cfg.anchor_shift_value
    anchor_shift_y = np.arange(0, score_height) * cfg.anchor_shift_value
    anchor_shift_x, anchor_shift_y = np.meshgrid(anchor_shift_x, anchor_shift_y)
    anchor_shifts = np.vstack((anchor_shift_x.ravel(), anchor_shift_y.ravel(), anchor_shift_x.ravel(), anchor_shift_y.ravel())).transpose()
    anchor_shift_num = anchor_shifts.shape[0]

    anchors = anchors.reshape((1, cfg.anchor_num, 4)) + anchor_shifts.reshape((1, anchor_shift_num, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((cfg.anchor_num * anchor_shift_num, 4))

    inside_anchors_idx = np.where((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < cfg.image_size_width) & (anchors[:, 3] < cfg.image_size_height))[0]
    anchors = anchors[inside_anchors_idx, :]

    labels = np.empty((len(inside_anchors_idx), ), dtype=np.float32)
    labels.fill(-1)

    overlaps = bo.overlap_bbox(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(bbox_holder, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inside_anchors_idx)), argmax_overlaps]

    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[max_overlaps < cfg.anchor_train_negative_overlap] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.anchor_train_positive_overlap] = 1

    num_fg = int(cfg.anchor_train_fg_ratio * cfg.anchor_train_batch_size)
    fg_idx = np.where(labels == 1)[0]
    if len(fg_idx) > num_fg:
        disable_idx = np.random.choice(fg_idx, size=(len(fg_idx) - num_fg), replace=False)
        labels[disable_idx] = -1

    num_bg = cfg.anchor_train_batch_size - np.sum(labels == 1)
    bg_idx = np.where(labels == 0)[0]
    if len(bg_idx) > num_bg:
        disable_idx = np.random.choice(bg_idx, size=(len(bg_idx) - num_bg), replace=False)
        labels[disable_idx] = -1

    argmax_overlap_bbox = bbox_holder[argmax_overlaps, :]
    bbox_targets = np.zeros((len(inside_anchors_idx), 4), dtype=np.float32)
    bbox_targets = bo.transform_bbox_train(anchors, argmax_overlap_bbox[:, :4]).astype(np.float32, copy=False)

    bbox_inside_weights = np.zeros((len(inside_anchors_idx), 4), dtype=np.float32)
    bbox_outside_weights = np.zeros((len(inside_anchors_idx), 4), dtype=np.float32)

    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples

    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    total_anchors = int(cfg.anchor_num * anchor_shift_num)
    labels = unmap_anchors(labels, total_anchors, inside_anchors_idx, fill=-1)
    bbox_targets = unmap_anchors(bbox_targets, total_anchors, inside_anchors_idx, fill=0)
    bbox_inside_weights = unmap_anchors(bbox_inside_weights, total_anchors, inside_anchors_idx, fill=0)
    bbox_outside_weights = unmap_anchors(bbox_outside_weights, total_anchors, inside_anchors_idx, fill=0)

    labels = labels.reshape((1, score_height, score_width, cfg.anchor_num)).transpose(0, 3, 1, 2)

    rpn_labels = labels.reshape((1, 1, cfg.anchor_num * score_height, score_width))
    rpn_bbox_targets = bbox_targets.reshape((1, score_height, score_width, cfg.anchor_num * 4)).transpose(0, 3, 1, 2)
    rpn_bbox_inside_weights = bbox_inside_weights.reshape((1, score_height, score_width, cfg.anchor_num * 4)).transpose(0, 3, 1, 2)
    rpn_bbox_outside_weights = bbox_outside_weights.reshape((1, score_height, score_width, cfg.anchor_num * 4)).transpose(0, 3, 1, 2)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
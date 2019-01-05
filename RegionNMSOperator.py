import numpy as np

import Configuration as cfg

def width_height_centers(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    cx = anchor[0] + 0.5 * (w - 1)
    cy = anchor[1] + 0.5 * (h - 1)
    return w, h, cx, cy

def make_anchors(ws, hs, cx, cy):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((cx - 0.5 * (ws - 1), cy - 0.5 * (hs - 1), cx + 0.5 * (ws - 1), cy + 0.5 * (hs - 1)))
    return anchors

def ratio_enum(anchor, ratios):
    w, h, cx, cy = width_height_centers(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = make_anchors(ws, hs, cx, cy)
    return anchors

def scale_enum(anchor, scales):
    w, h, cx, cy = width_height_centers(anchor)
    ws = w * scales
    hs = h * scales
    anchors = make_anchors(ws, hs, cx, cy)
    return anchors

def bbox_transform_detect(anchors, preds):
    if anchors.shape[0] == 0:
        return np.zeros((0, preds.shape[1]), dtype=preds.dtype)

    anchors = anchors.astype(preds.dtype, copy=False)

    w = anchors[:, 2] - anchors[:, 0] + 1.0
    h = anchors[:, 3] - anchors[:, 1] + 1.0
    cx = anchors[:, 0] + 0.5 * w
    cy = anchors[:, 1] + 0.5 * h

    dx = preds[:, 0::4]
    dy = preds[:, 1::4]
    dw = preds[:, 2::4]
    dh = preds[:, 3::4]

    pred_cx = dx * w[:, np.newaxis] + cx[:, np.newaxis]
    pred_cy = dy * h[:, np.newaxis] + cy[:, np.newaxis]
    pred_w = np.exp(dw) * w[:, np.newaxis]
    pred_h = np.exp(dh) * h[:, np.newaxis]

    pred_bbox = np.zeros(preds.shape, dtype=preds.dtype)
    pred_bbox[:, 0::4] = pred_cx - 0.5 * pred_w
    pred_bbox[:, 1::4] = pred_cy - 0.5 * pred_h
    pred_bbox[:, 2::4] = pred_cx + 0.5 * pred_w
    pred_bbox[:, 3::4] = pred_cy + 0.5 * pred_h

    return pred_bbox

def clip_bbox(bbox):
    bbox[:, 0::4] = np.maximum(np.minimum(bbox[:, 0::4], cfg.image_size_width - 1), 0)
    bbox[:, 1::4] = np.maximum(np.minimum(bbox[:, 1::4], cfg.image_size_height - 1), 0)
    bbox[:, 2::4] = np.maximum(np.minimum(bbox[:, 2::4], cfg.image_size_width - 1), 0)
    bbox[:, 3::4] = np.maximum(np.minimum(bbox[:, 3::4], cfg.image_size_height - 1), 0)
    return bbox

def filter_bbox(bbox, min_size):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def nms(bbox, nms_thresh):
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    probs = bbox[:, 4]
    order = probs.argsort()[::-1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    bbox_num = bbox.shape[0]
    suppressed = np.zeros(bbox_num, dtype=np.int)

    keep = []
    for i in range(bbox_num):
        bbox_idx1 = order[i]
        if suppressed[bbox_idx1] == 1:
            continue

        keep.append(bbox_idx1)
        for j in range(i + 1, bbox_num):
            bbox_idx2 = order[j]
            if suppressed[bbox_idx2] == 1:
                continue

            inter_x1 = max(x1[bbox_idx1], x1[bbox_idx2])
            inter_y1 = max(y1[bbox_idx1], y1[bbox_idx2])
            inter_x2 = min(x2[bbox_idx1], x2[bbox_idx2])
            inter_y2 = min(y2[bbox_idx1], y2[bbox_idx2])
            inter_w = max(0.0, inter_x2 - inter_x1 + 1)
            inter_h = max(0.0, inter_y2 - inter_y1 + 1)
            inter_area = inter_w * inter_h

            inter_ratio = inter_area / (areas[bbox_idx1] + areas[bbox_idx2] - inter_area)
            if inter_ratio >= nms_thresh:
                suppressed[bbox_idx2] = 1

    return keep

def nms_region(rpn_cls_prob, rpn_bbox_pred, trainable):
    base_anchor = np.array([1, 1, cfg.anchor_base_size, cfg.anchor_base_size]) - 1
    ratio_anchors = ratio_enum(base_anchor, cfg.anchor_ratios)
    anchors = np.vstack([scale_enum(ratio_anchors[i], np.array(cfg.anchor_scales)) for i in range(ratio_anchors.shape[0])])

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

    proposals = bbox_transform_detect(anchors, bbox_preds)
    proposals = clip_bbox(proposals)

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

    keep = filter_bbox(proposals, min_size)
    proposals = proposals[keep, :]
    cls_probs = cls_probs[keep]

    order = cls_probs.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    cls_probs = cls_probs[order]
    
    keep = nms(np.hstack((proposals, cls_probs)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    cls_probs = cls_probs[keep]

    batch_ids = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    nms_region = np.hstack((batch_ids, proposals.astype(np.float32, copy=False)))
    return nms_region
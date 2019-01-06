import numpy as np

import Configuration as cfg

def transform_bbox_detect(input_bbox, pred_bbox):
    if input_bbox.shape[0] == 0:
        return np.zeros((0, pred_bbox.shape[1]), dtype=pred_bbox.dtype)

    input_bbox = input_bbox.astype(pred_bbox.dtype, copy=False)

    iw = input_bbox[:, 2] - input_bbox[:, 0] + 1.0
    ih = input_bbox[:, 3] - input_bbox[:, 1] + 1.0
    icx = input_bbox[:, 0] + 0.5 * iw
    icy = input_bbox[:, 1] + 0.5 * ih

    px = pred_bbox[:, 0::4]
    py = pred_bbox[:, 1::4]
    pw = pred_bbox[:, 2::4]
    ph = pred_bbox[:, 3::4]

    cx = px * iw[:, np.newaxis] + icx[:, np.newaxis]
    cy = py * ih[:, np.newaxis] + icy[:, np.newaxis]
    w = np.exp(pw) * iw[:, np.newaxis]
    h = np.exp(ph) * ih[:, np.newaxis]

    bbox = np.zeros(pred_bbox.shape, dtype=pred_bbox.dtype)
    bbox[:, 0::4] = cx - 0.5 * w
    bbox[:, 1::4] = cy - 0.5 * h
    bbox[:, 2::4] = cx + 0.5 * w
    bbox[:, 3::4] = cy + 0.5 * h

    return bbox

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

def nms_bbox(bbox, nms_thresh):
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    probs = bbox[:, 4]
    order = probs.argsort()[::-1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    bbox_num = bbox.shape[0]
    suppressed = np.zeros(bbox_num, dtype=np.bool)

    keep = []
    for i in range(bbox_num):
        bbox_idx1 = order[i]
        if suppressed[bbox_idx1]:
            continue

        keep.append(bbox_idx1)
        for j in range(i + 1, bbox_num):
            bbox_idx2 = order[j]
            if suppressed[bbox_idx2]:
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
                suppressed[bbox_idx2] = True

    return keep

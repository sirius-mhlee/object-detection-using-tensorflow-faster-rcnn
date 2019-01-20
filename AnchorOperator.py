import numpy as np

import Configuration as cfg

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
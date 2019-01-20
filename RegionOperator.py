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

    if trainable:
        return roi
    else:
        return nms_region
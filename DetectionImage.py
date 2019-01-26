import sys
import cv2
import os
import random as rand

import numpy as np
import tensorflow as tf

import Configuration as cfg

import AlexNetConv as anc
import RegionProposalNetwork as rpn
import DetectionNetwork as dn

import DataOperator as do
import BBoxOperator as bo

def generate_image(label_file_path, img, nms_detect_list):
    label_file = open(label_file_path, 'r')
    synset = [line.strip() for line in label_file.readlines()]
    label_file.close()

    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(len(synset))]

    save_img = img.copy()
    height, width, channel = save_img.shape

    for detect in nms_detect_list:
        left = int(max(detect[2], 0))
        top = int(max(detect[3], 0))
        right = int(min(detect[4], width))
        bottom = int(min(detect[5], height))

        cv2.rectangle(save_img, (left, top), (right, bottom), color[detect[0]], 2)

        text_size, baseline = cv2.getTextSize(' ' + synset[detect[0]] + ' ', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(save_img, (left, top - text_size[1] - (baseline * 2)), (left + text_size[0], top), color[detect[0]], -1)
        cv2.putText(save_img, ' ' + synset[detect[0]] + ' ', (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return save_img

def main():
    with tf.Session() as sess:
        # (Image) -> AlexNetConv -> (Feature Image)
        # (Feature Image) -> RegionProposalNetwork -> (ROI Coord List with Prob)
        # (ROI Coord List with Prob) -> RegionNMS -> (ROI Coord that deleted small ROI, and low Prob ROI)
        # (Feature Image), (ROI Coord that deleted small ROI, and low Prob ROI) -> DetectionNetwork -> (Final BBOX), (Class Prob)

        image = tf.placeholder(tf.float32, [1, cfg.image_size_width, cfg.image_size_height, 3])
        feature = tf.placeholder(tf.float32, [1, 6, 6, 256])
        rpn_cls_prob = tf.placeholder(tf.float32, [1, 6, 6, 256])
        rpn_bbox_pred = tf.placeholder(tf.float32, [1, 6, 6, 256])

        model = do.load_model(sys.argv[1])
        mean = do.load_mean(sys.argv[2])
        alexnetconv_model = anc.AlexNetConv(model, mean, False)
        with tf.name_scope('alexnetconv_content'):
            alexnetconv_model.build(image)

        model = do.load_model(sys.argv[3])
        rpn_model = rpn.RegionProposalNetwork(model, False)
        with tf.name_scope('rpn_content'):
            rpn_model.build(feature)

        model = do.load_model(sys.argv[4])
        detection_model = dn.DetectionNetwork(model, False)
        with tf.name_scope('detection_content'):
            detection_model.build(feature, rpn_cls_prob, rpn_bbox_pred)

        sess.run(tf.global_variables_initializer())

        img, expand_np_img, width, height = do.load_image(sys.argv[6])

        region_scale_width = cfg.image_size_width / width
        region_scale_height = cfg.image_size_height / height

        feed_dict = {image:expand_np_img}
        conv_feature = sess.run([alexnetconv_model.pool5], feed_dict=feed_dict)

        feed_dict = {feature:conv_feature[0]}
        cls_prob, bbox_pred = sess.run([rpn_model.rpn_cls_prob, rpn_model.rpn_bbox_pred], feed_dict=feed_dict)

        feed_dict = {feature:conv_feature[0], rpn_cls_prob:cls_prob, rpn_bbox_pred:bbox_pred}
        region_prob, region_bbox = sess.run([detection_model.cls_prob, detection_model.bbox_pred], feed_dict=feed_dict)

        region_bbox = bo.transform_bbox_detect(nms_region[:, 1:], region_bbox)
        region_bbox = bo.clip_bbox(region_bbox)

        detect_list = []
        for i in range(0, cfg.object_class_num):
            idx = np.where(region_prob[:, i] > cfg.detect_prob_thresh)[0]
            if len(idx) > 0:
                prob = region_prob[idx, i]
                bbox = region_bbox[idx, i * 4:(i + 1) * 4]

                region_list = np.hstack((bbox, prob[:, np.newaxis])).astype(np.float32, copy=False)
                keep = bo.nms_bbox(region_list, cfg.detect_nms_thresh)
                region_list = region_list[keep, :]

                for detect in region_list:
                    x1 = detect[0] / region_scale_width
                    y1 = detect[1] / region_scale_height
                    x2 = detect[2] / region_scale_width
                    y2 = detect[3] / region_scale_height
                    detect_list.append((label, detect[4], x1, y1, x2, y2))

        save_img = generate_image(sys.argv[5], img, detect_list)
        cv2.imwrite(sys.argv[7], save_img)

if __name__ == '__main__':
    main()

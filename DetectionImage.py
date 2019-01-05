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
import RegionNMSOperator as nmso

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
        region = tf.placeholder(tf.float32, [None, 5])

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
            detection_model.build(feature, region)

        sess.run(tf.global_variables_initializer())

        expand_np_img, width, height = do.load_image(sys.argv[6])

        region_scale_width = cfg.image_size_width / width
        region_scale_height = cfg.image_size_height / height

        feed_dict = {image:expand_np_img}
        conv_feature = sess.run([alexnetconv_model.pool5], feed_dict=feed_dict)

        feed_dict = {feature:conv_feature[0]}
        cls_prob, bbox_pred = sess.run([rpn_model.rpn_cls_prob, rpn_model.rpn_bbox_pred], feed_dict=feed_dict)

        nms_region = nmso.nms_region(cls_prob, bbox_pred, False)

        feed_dict = {feature:conv_feature[0], region:nms_region}
        region_prob, region_bbox = sess.run([detection_model.cls_prob, detection_model.bbox_pred], feed_dict=feed_dict)

        #region_bbox = bbox_transform_detect(nms_region[:, 1:], region_bbox)
        #region_bbox = clip_bbox(region_bbox)

        #detect_list = []
        #for j in range(1, cfg.NUM_CLASSES):
        #    inds = np.where(region_prob[:, j] > thresh)[0]
        #    if len(inds) == 0:
        #        continue
        #    cls_probs = region_prob[inds, j]                  # Class Probabilities
        #    cls_boxes = region_bbox[inds, j * 4:(j + 1) * 4]  # Class Box Predictions
        #    cls_dets = np.hstack((cls_boxes, cls_probs[:, np.newaxis])) \
        #        .astype(np.float32, copy=False)
        #    keep = nms(cls_dets, cfg.TEST.NMS)          # Apply NMS
        #    cls_dets = cls_dets[keep, :]
        #    detect_list.append(cls_dets)

        #for i in range(len(region_prob)):
        #    prob = np.max(region_prob[i])
        #    label = np.argmax(region_prob[i])
        #    if label != cfg.object_class_num and prob > 0.5:
        #        region = anchor[i]
        #        region_width = (region.rect.right * region_scale_width) - (region.rect.left * region_scale_width)
        #        region_hegith = (region.rect.bottom * region_scale_height) - (region.rect.top * region_scale_height)
        #        region_center_x = (region.rect.left * region_scale_width) + region_width / 2
        #        region_center_y = (region.rect.top * region_scale_height) + region_hegith / 2

        #        bbox_center_x = region_width * region_bbox[i][(label * 4) + 0] + region_center_x
        #        bbox_center_y = region_hegith * region_bbox[i][(label * 4) + 1] + region_center_y
        #        bbox_width = region_width * np.exp(region_bbox[i][(label * 4) + 2])
        #        bbox_height = region_hegith * np.exp(region_bbox[i][(label * 4) + 3])

        #        bbox_left = bbox_center_x - bbox_width / 2
        #        bbox_top = bbox_center_y - bbox_height / 2
        #        bbox_right = bbox_center_x + bbox_width / 2
        #        bbox_bottom = bbox_center_y + bbox_height / 2

        #        detect_list.append((label, region_prob[i][label], bbox_left / region_scale_width, bbox_top / region_scale_height, bbox_right / region_scale_width, bbox_bottom / region_scale_height))

        #save_img = generate_image(sys.argv[5], img, detect_list)
        #cv2.imwrite(sys.argv[7], save_img)

if __name__ == '__main__':
    main()

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
import RegionOperator as ro

def print_epoch_info(epoch_idx, loss_value):
    print('Epoch : {0}, Loss Sum : {1}'.format(epoch_idx, loss_value))

def main():
    with tf.Session() as sess:
        train_data = do.load_train_data(sys.argv[3])
        train_size = len(train_data)

        image = tf.placeholder(tf.float32, [1, cfg.image_size_width, cfg.image_size_height, 3])
        region = tf.placeholder(tf.float32, [None, 5])

        model = do.load_model(sys.argv[1])
        mean = do.load_mean(sys.argv[2])
        alexnetconv_model = anc.AlexNetConv(model, mean)
        with tf.name_scope('alexnetconv_content'):
            alexnetconv_model.build(image)

        rpn_model = rpn.RegionProposalNetwork(None, True)
        with tf.name_scope('rpn_content'):
            rpn_model.build(alexnetconv_model.pool5, region)

        detection_model = dn.DetectionNetwork(None, True)
        with tf.name_scope('detection_content'):
            detection_model.build(alexnetconv_model.pool5, rpn_model.rpn_cls_prob, rpn_model.rpn_bbox_pred, region)
    
        step_cnt = 0
        with tf.name_scope('total_loss'):
            loss = tf.reduce_sum(rpn_model.rpn_cls_loss + rpn_model.rpn_bbox_loss + detection_model.detection_cls_loss + (1.0 * detection_model.detection_region_loss))

            decay_steps = cfg.learning_rate_decay_ratio * train_size
            learning_rate = tf.train.exponential_decay(learning_rate=cfg.learning_rate, global_step=step_cnt, decay_steps=decay_steps, decay_rate=cfg.learning_rate_decay_factor, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        writer = tf.summary.FileWriter('./log/', sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Training Model')
        for epoch_idx in range(cfg.training_max_epoch):
            step_cnt += 1

            train_image, train_bbox = do.get_train_data(sess, train_data)

            feed_dict = {image:train_image, region:train_bbox}
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)

            print_epoch_info(epoch_idx, loss_value)

        do.save_model(sess, rpn_model.var_dict, sys.argv[4])
        do.save_model(sess, detection_model.var_dict, sys.argv[5])

if __name__ == '__main__':
    main()

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

def print_batch_info(epoch_idx, batch_idx, loss_mean_value):
    print('Epoch : {0}, Batch : {1}, Loss Mean : {2}'.format(epoch_idx, batch_idx, loss_mean_value))

def print_epoch_info(epoch_idx, accuracy_mean_value):
    print('Epoch : {0}, Accuracy Mean : {1}'.format(epoch_idx, accuracy_mean_value))

def main():
    with tf.Session() as sess:
        alexnet_train_data, alexnet_train_mean = do.load_alexnet_train_data(sys.argv[1])
        alexnet_train_size = len(alexnet_train_data)

        alexnet_finetune_data = do.load_alexnet_finetune_data(sys.argv[2])
        alexnet_finetune_size = len(alexnet_finetune_data)

        # (Image) -> AlexNetConv -> (Feature Image)
        # (Feature Image) -> RegionProposalNetwork -> (ROI Coord List with Prob)
        # (ROI Coord List with Prob) -> RegionNMS -> (ROI Coord that deleted small ROI, and low Prob ROI)
        # (Feature Image), (ROI Coord that deleted small ROI, and low Prob ROI) -> DetectionNetwork -> (Final BBOX), (Class Prob)

        image = tf.placeholder(tf.float32, [1, cfg.image_size_width, cfg.image_size_height, 3])
        feature = tf.placeholder(tf.float32, [1, 6, 6, 256])
        region = tf.placeholder(tf.float32, [None, 5])

        alexnetconv_model = anc.AlexNetConv(None, mean, True)
        with tf.name_scope('alexnetconv_content'):
            alexnetconv_model.build(image)

        rpn_model = rpn.RegionProposalNetwork(None, True)
        with tf.name_scope('rpn_content'):
            rpn_model.build(feature)

        detection_model = dn.DetectionNetwork(None, True)
        with tf.name_scope('detection_content'):
            detection_model.build(feature, region)

        writer = tf.summary.FileWriter('./log/', sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Training AlexNet')
        for epoch_idx in range(cfg.training_max_epoch):
            for batch_idx in range(alexnet_train_size // cfg.batch_size):
                batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, cfg.batch_size)
                feed_dict = {image:batch_image, label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.optimizer, alexnet_model.loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, cfg.batch_size)
            feed_dict = {image:batch_image, label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        print('Finetuning AlexNet')
        for epoch_idx in range(cfg.finetuning_max_epoch):
            for batch_idx in range(alexnet_finetune_size // cfg.batch_size):
                batch_image, batch_bbox, batch_bbox_slice_idx, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, cfg.batch_size)
                feed_dict = {image:batch_image, bbox:batch_bbox, bbox_slice_idx:batch_bbox_slice_idx, finetune_label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.finetune_optimizer, alexnet_model.finetune_loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_bbox, batch_bbox_slice_idx, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, cfg.batch_size)
            feed_dict = {image:batch_image, bbox:batch_bbox, bbox_slice_idx:batch_bbox_slice_idx, finetune_label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.finetune_accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        do.save_model(sess, alexnet_model.var_dict, sys.argv[3])
        do.save_mean(alexnet_model.mean, sys.argv[4])

if __name__ == '__main__':
    main()

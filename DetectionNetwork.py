import numpy as np
import tensorflow as tf

import Configuration as cfg

import RegionOperator as ro

class DetectionNetwork:
    def __init__(self, model, trainable):
        self.model = model
        self.var_dict = {}
        self.trainable = trainable

    def build(self, feature_holder, cls_prob_holder, bbox_pred_holder, region_holder=None):
        self.nms_region = tf.reshape(tf.py_func(ro.nms_region, [cls_prob_holder, bbox_pred_holder, self.trainable], [tf.float32]), [-1, 5])
        reorder_nms_region = tf.stack([self.nms_region[:,0], self.nms_region[:,2], self.nms_region[:,1], self.nms_region[:,4], self.nms_region[:,3]], axis=1)

        if self.trainable:
            rois, labels, region_targets, region_inside_weights, region_outside_weights = tf.py_func(ro.target_region, [self.nms_region, region_holder], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            self.rois = tf.reshape(rois, [-1, 5]) 
            self.labels = tf.convert_to_tensor(tf.cast(labels, dtype=tf.int32))
            self.region_targets = tf.convert_to_tensor(region_targets)
            self.region_inside_weights = tf.convert_to_tensor(region_inside_weights)
            self.region_outside_weights = tf.convert_to_tensor(region_outside_weights)

            reorder_nms_region = tf.stack([self.rois[:,0], self.rois[:,2], self.rois[:,1], self.rois[:,4], self.rois[:,3]], axis=1)

        self.feature_roi = self.roi_pool(feature_holder, reorder_nms_region, 14, 'feature_roi')
        self.feature_pool = self.max_pool(self.feature_roi, 2, 2, 'SAME', 'feature_pool')

        self.fc1 = self.fc_layer(self.feature_pool, 12544, 1024, 'fc1')
        self.relu1 = tf.nn.relu(self.fc1)

        if self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, 0.5)

        self.fc2 = self.fc_layer(self.relu1, 1024, 1024, 'fc2')
        self.relu2 = tf.nn.relu(self.fc2)

        if self.trainable:
            self.relu2 = tf.nn.dropout(self.relu2, 0.5)

        self.cls_fc = self.fc_layer(self.relu2, 1024, cfg.object_class_num, 'cls_fc')
        self.cls_prob = tf.nn.softmax(self.cls_fc, name='cls_prob')

        self.bbox_pred = self.fc_layer(self.relu2, 1024, cfg.object_class_num * 4, 'bbox_pred')

        if self.trainable:
            with tf.name_scope('detection_cls_loss'):
                self.detection_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(self.cls_fc), labels=self.labels))

            with tf.name_scope('detection_bbox_loss'):
                diff = tf.multiply(self.region_inside_weights, self.bbox_pred - self.region_targets)

                sigma = 1.0
                conditional = tf.less(tf.abs(diff), 1 / sigma ** 2)
                close = 0.5 * (sigma * x) ** 2
                far = tf.abs(x) - 0.5 / sigma ** 2
                diff_smooth_L1 = tf.where(conditional, close, far)
        
                self.detection_region_loss = 1.0 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.region_outside_weights, diff_smooth_L1), reduction_indices=[1]))

    def get_var(self, initial_value, name, idx, var_name):
        if self.model is not None and name in self.model:
            value = self.model[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + '_weights')

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + '_biases')

        return weights, biases

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def max_pool(self, bottom, kernel_size, stride_size, padding_type, name):
        return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding_type, name=name)

    def roi_pool(self, bottom, box_holder, crop_size, name):
        box_rect = box_holder[:, 1:5] / [cfg.image_size_height, cfg.image_size_width, cfg.image_size_height, cfg.image_size_width]
        box_batch_idx = tf.cast(box_holder[:, 0], tf.int32)
        return tf.image.crop_and_resize(bottom, box_rect, box_batch_idx, [crop_size, crop_size], name=name)


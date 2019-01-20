import numpy as np
import tensorflow as tf

import Configuration as cfg

import AnchorOperator as ao

class RegionProposalNetwork:
    def __init__(self, model, trainable):
        self.model = model
        self.var_dict = {}
        self.trainable = trainable

    def build(self, feature_holder, label_holder=None):
        self.rpn_conv1 = self.conv_layer(feature_holder, 256, 256, 3, 1, 'SAME', True, 'rpn_conv1')

        self.rpn_cls_score = self.conv_layer(self.rpn_conv1, 256, cfg.anchor_num * 2, 1, 1, 'SAME', False, 'rpn_cls_score')
        self.rpn_bbox_pred = self.conv_layer(self.rpn_conv1, 256, cfg.anchor_num * 4, 1, 1, 'SAME', False, 'rpn_bbox_pred')

        if self.trainable:
            self.anchor_target()
            self.rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            self.rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            self.rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            self.rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')

        #    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=label_holder)
        #    self.loss_mean = tf.reduce_mean(self.loss)
        #    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(self.loss_mean)

        #    self.correct_prediction = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(label_holder, 1))
        #    self.accuracy_mean = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        rpn_cls_shape = tf.shape(self.rpn_cls_score)

        self.rpn_cls_score = tf.transpose(self.rpn_cls_score, [0, 3, 1, 2])
        self.rpn_cls_score = tf.reshape(self.rpn_cls_score, [rpn_cls_shape[0], 2, rpn_cls_shape[3] // 2 * rpn_cls_shape[1], rpn_cls_shape[2]])
        self.rpn_cls_score = tf.transpose(self.rpn_cls_score, [0, 2, 3, 1])

        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score)

        self.rpn_cls_prob = tf.transpose(self.rpn_cls_prob, [0, 3, 1, 2])
        self.rpn_cls_prob = tf.reshape(self.rpn_cls_prob, [rpn_cls_shape[0], rpn_cls_shape[3], rpn_cls_shape[1], rpn_cls_shape[2]])
        self.rpn_cls_prob = tf.transpose(self.rpn_cls_prob, [0, 2, 3, 1])

    def get_var(self, initial_value, name, idx, var_name):
        if self.model is not None and name in self.model:
            value = self.model[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + '_filters')

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + '_biases')

        return filters, biases

    def conv_layer(self, bottom, in_channels, out_channels, filter_size, stride_size, padding_type, relu_activate, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filter=filt, strides=[1, stride_size, stride_size, 1], padding=padding_type)
            bias = tf.nn.bias_add(conv, conv_biases)
            if relu_activate == False:
                return bias

            relu = tf.nn.relu(bias)

            return relu

    def anchor_target(self):
        _anchors = ao.generate_anchors()
        return None

    #def get_var_count(self):
    #    count = 0
    #    for var in list(self.var_dict.values()):
    #        count += np.multiply(var.get_shape().as_list())
    #    return count

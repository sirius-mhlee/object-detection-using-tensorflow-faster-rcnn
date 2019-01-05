import numpy as np
import tensorflow as tf

import Configuration as cfg

class DetectionNetwork:
    def __init__(self, model, trainable):
        self.model = model
        self.var_dict = {}
        self.trainable = trainable

    def build(self, feature_holder, region_holder, label_holder=None):
        region_holder = tf.stack([region_holder[:,0], region_holder[:,2], region_holder[:,1], region_holder[:,4], region_holder[:,3]], axis=1)

        self.feature_roi = self.roi_pool(feature_holder, region_holder, 14, 'feature_roi')
        self.feature_pool = self.max_pool(self.feature_roi, 2, 2, 'SAME', 'feature_pool')

        self.fc1 = self.fc_layer(self.feature_pool, 12544, 1024, 'fc1')
        self.relu1 = tf.nn.relu(self.fc1)

        if self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, 0.5)

        self.fc2 = self.fc_layer(self.relu1, 1024, 1024, 'fc2')
        self.relu2 = tf.nn.relu(self.fc2)

        if self.trainable:
            self.relu2 = tf.nn.dropout(self.relu2, 0.5)

        self.cls_prob = self.fc_layer(self.relu2, 1024, cfg.object_class_num, 'cls_prob')
        self.bbox_pred = self.fc_layer(self.relu2, 1024, cfg.object_class_num * 4, 'bbox_pred')

        #if self.trainable:
        #    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=label_holder)
        #    self.loss_mean = tf.reduce_mean(self.loss)
        #    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(self.loss_mean)

        #    self.correct_prediction = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(label_holder, 1))
        #    self.accuracy_mean = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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

    #def get_var_count(self):
    #    count = 0
    #    for var in list(self.var_dict.values()):
    #        count += np.multiply(var.get_shape().as_list())
    #    return count

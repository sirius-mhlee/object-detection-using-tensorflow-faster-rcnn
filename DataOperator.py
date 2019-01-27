import cv2
import random as rand

import numpy as np
import tensorflow as tf

import Configuration as cfg

def load_model(model_path):
    model = np.load(model_path, encoding='latin1').item()
    return model

def save_model(sess, var_dict, model_path):
    data_dict = {}

    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name][idx] = var_out

    np.save(model_path, data_dict)

def load_mean(mean_path):
    mean_file = open(mean_path, 'r')
    line = mean_file.readline()
    split_line = line.split(' ')
    mean = [float(split_line[0]), float(split_line[1]), float(split_line[2])]
    mean_file.close()
    return mean

def save_mean(mean, mean_path):
    mean_file = open(mean_path, 'w')
    mean_file.write('{0} {1} {2}'.format(mean[0], mean[1], mean[2]))
    mean_file.close()

def load_image(img_path):
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    reshape_img = cv2.resize(img, dsize=(cfg.image_size_width, cfg.image_size_height), interpolation=cv2.INTER_CUBIC)
    np_img = np.asarray(reshape_img, dtype=float)
    expand_np_img = np.expand_dims(np_img, axis=0)
    return img, expand_np_img, width, height
    
def load_train_data(train_data_path):
    train_data = []

    train_file = open(train_data_path, 'r')
    all_line = train_file.readlines()
    for line in all_line:
        split_line = line.split(' ')
        train_data.append((split_line[0], int(split_line[1]), int(split_line[2]), int(split_line[3]), int(split_line[2]) + int(split_line[4]), int(split_line[3]) + int(split_line[5])))
    train_file.close()

    return train_data

def get_train_data(sess, train_data):    
    rand.shuffle(train_data)

    image = []
    bbox = []

    for data in train_data:
        _, expand_np_img, width, height = load_image(data[0])
        image.append(expand_np_img)

        region_scale_width = cfg.image_size_width / width
        region_scale_height = cfg.image_size_height / height

        bbox.append((data[2] * region_scale_width, data[3] * region_scale_height, data[4] * region_scale_width, data[5] * region_scale_height, data[1]))

    batch_image = np.concatenate(image)
    batch_bbox_op = tf.convert_to_tensor(bbox, dtype=tf.float32)
    batch_bbox = sess.run(batch_bbox_op)

    return batch_image, batch_bbox

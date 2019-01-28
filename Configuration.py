image_size_width = 227
image_size_height = 227
object_class_num = 17
object_class_num_with_bg = 18

training_max_epoch = 100
learning_rate = 0.000005
learning_rate_decay_factor = 0.5
learning_rate_decay_ratio = 10

detect_prob_thresh = 0.5
detect_nms_thresh = 0.01
detect_size_thresh = 16

anchor_num = 9
anchor_base_size = 16
anchor_ratios = [0.5, 1.0, 2.0]
anchor_scales = [2, 4, 8]

anchor_pre_nms_topN_train = 12000
anchor_post_nms_topN_train = 2000
anchor_nms_thresh_train = 0.7
anchor_min_size_train = 16

anchor_pre_nms_topN_detect = 6000
anchor_post_nms_topN_detect = 300
anchor_nms_thresh_detect = 0.7
anchor_min_size_detect = 16

anchor_shift_value = 12

anchor_train_batch_size = 256
anchor_train_positive_overlap = 0.7
anchor_train_negative_overlap = 0.3
anchor_train_fg_ratio = 0.5

region_train_batch_size = 128
region_train_fg_ratio = 0.25
region_train_fg_thresh = 0.5
region_train_bg_max_ratio = 0.02
region_train_bg_thresh_lo = 0.0
region_train_bg_thresh_hi = 0.5
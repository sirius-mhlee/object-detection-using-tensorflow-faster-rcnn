image_size_width = 227
image_size_height = 227
object_class_num = 17

#training_max_epoch = 1
#finetuning_max_epoch = 3
#batch_size = 1
#region_per_batch = 3

detect_prob_thresh = 0.5
detect_nms_thresh = 0.5

anchor_num = 9
anchor_base_size = 16
anchor_ratios = [0.5, 1.0, 2.0]
anchor_scales = [8, 16, 32]

anchor_pre_nms_topN_detect = 6000
anchor_post_nms_topN_detect = 300
anchor_nms_thresh_detect = 0.7
anchor_min_size_detect = 16

anchor_shift_value = 12

from __future__ import division
import os
import tensorflow as tf
import numpy as np

class GleasonConfig():
    def __init__(self, **kwargs):

        # directories for storing tfrecords, checkpoints etc.
        self.main_dir = '/media/data_cifs/andreas/pathology/gleason_training/'

        self.checkpoint_path = os.path.join(self.main_dir, 'checkpoints')
        self.summary_path = os.path.join(self.main_dir, 'summaries')
        self.results_path = os.path.join(self.main_dir, 'results')
       
        self.train_fn =  os.path.join(self.main_dir, 'tfrecords/train2.tfrecords')
        self.val_fn =  os.path.join(self.main_dir, 'tfrecords/val2.tfrecords')
        self.test_fn =  os.path.join(self.main_dir, 'tfrecords/test2.tfrecords')

        self.exp_fn = os.path.join(self.main_dir, 'tfrecords/exp.tfrecords')

        # self.test_checkpoint = os.path.join(self.checkpoint_path,'unet/unet_cross_entropy_2018_08_01_12_02_23_6800.ckpt')
        # self.test_checkpoint = os.path.join(self.checkpoint_path,'unet/unet_focal_loss_2018_08_01_14_10_19_239400.ckpt') 
        self.test_checkpoint = os.path.join(self.checkpoint_path,'unet/unet_sigmoid_cross_entropy_2018_08_03_07_24_34_1900.ckpt') 
        
        self.optimizer = "adam"
        self.momentum = 0.9 # if optimizer is nestrov

        self.initial_learning_rate = 3e-04
        self.decay_learning_rate = True
        self.decay_steps = 5000 # number of steps before decaying the learning rate
        self.learning_rate_decay_factor = 0.5 
        
        self.train_batch_size = 2
        self.val_batch_size = 2

        self.num_batches_to_validate_over = 50 # number of batches to validate over 32*100 = 3200
        self.validate_every_num_steps = 100 # perform a validation step

        self.num_train_epochs = 10000
        self.input_image_size = [512, 512, 3] # size of the input tf record image
        
        self.train_augmentations_dic = {
                                        'rand_flip_left_right':True,
                                        'rand_flip_top_bottom':True,
                                        'rand_rotate':True,
                                        'warp':True,
                                        'distort_brightness_constrast':True,
                                        'grayscale':False
                                       }

        self.val_augmentations_dic = {
                                      'rand_flip_left_right':False,
                                      'rand_flip_top_bottom':False,
                                      'rand_rotate':False,
                                      'warp':False,
                                      'color_distor':False,
                                      'distort_brightness_constrast':False,
                                      'grayscale':False
                                     }

    def weighted_sigmoid_with_logits():
        # tf.nn.sigmoid_cross_entropy_with_logits
        pass

    def weighted_cross_entropy():
        pass

    def focal_loss(self, onehot_labels, logits, alpha=0.25, gamma=2.0, name=None, scope=None):
      
        with tf.name_scope(scope, 'focal_loss', [logits, onehot_labels]) as sc:


            predictions = tf.nn.sigmoid(logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
            # add small value to avoid 0
            epsilon = 1e-8
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
            losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                         name=name, axis=1)
            return losses

    def dice_loss():
        pass

    def rescale(self, tensor, min_new, max_new):

        min_old = tf.reduce_min(tensor)
        max_old = tf.reduce_max(tensor)

        ratio = (max_new - min_new) / ((max_old - min_old)+ 1e-8)
        tensor = tensor - max_old
        tensor = ratio*tensor
        tensor = tensor+max_new

        return tensor

    def pixel_accuracy(self, y_true, y_pred, num_clas = 5):

        flatten_pred = tf.reshape(y_pred, [-1])
        flatten_true = tf.reshape(y_true, [-1])

        con = tf.concat([flatten_true, flatten_pred], axis=-1) 
        uni_size = tf.cast(tf.size(tf.unique(con)[0]), dtype=tf.float32)

        c = tf.cast(tf.confusion_matrix(flatten_true, flatten_pred, num_classes=num_clas), dtype=tf.float32)
        intersection = tf.diag_part(c)

        total_pixels = tf.cast(tf.shape(flatten_true)[0], dtype=tf.float32)
        numerator =  tf.reduce_sum(intersection)
        pixel_acc = numerator / total_pixels

        return pixel_acc

    def mean_accuracy(self, y_true, y_pred, num_clas = 5):

        flatten_pred = tf.reshape(y_pred, [-1])
        flatten_true = tf.reshape(y_true, [-1])

        con = tf.concat([flatten_true, flatten_pred], axis=-1) 
        uni_size = tf.cast(tf.size(tf.unique(con)[0]), dtype=tf.float32)

        c = tf.cast(tf.confusion_matrix(flatten_true, flatten_pred, num_classes=num_clas), dtype=tf.float32)
        intersection = tf.diag_part(c)

        keys = tf.range(num_clas)
        sorted_inter = tf.gather(intersection, tf.nn.top_k(keys, k=num_clas).indices)

        mask = tf.concat([tf.cast(flatten_true, dtype=tf.int32), keys], axis=-1)
        mask_classes, _, mask_class_weights = tf.unique_with_counts(mask)
        reordered_mask_class_weights = tf.gather(mask_class_weights, tf.nn.top_k(mask_classes, k=num_clas).indices)
        bool_mask = tf.not_equal(reordered_mask_class_weights, [1])

        classes, _, class_weights = tf.unique_with_counts(flatten_true)
        true_class_size = tf.cast(tf.size(tf.unique(flatten_true)[0]), dtype=tf.int32)
        denominator = tf.cast(tf.gather(class_weights, tf.nn.top_k(classes, k=true_class_size).indices), dtype=tf.float32)  

        numerator = tf.boolean_mask(sorted_inter, bool_mask)
        mean_acc = (1/uni_size) * tf.reduce_sum((numerator / denominator))

        return mean_acc

    def mean_IOU(self, y_true, y_pred, num_clas = 5):

        flatten_pred = tf.reshape(y_pred, [-1])
        flatten_true = tf.reshape(y_true, [-1])

        con = tf.concat([flatten_true, flatten_pred], axis=-1) 
        uni_size = tf.cast(tf.size(tf.unique(con)[0]), dtype=tf.float32)

        c = tf.cast(tf.confusion_matrix(flatten_true, flatten_pred, num_classes=num_clas), dtype=tf.float32)
        intersection = tf.diag_part(c)
        ground_truth_set = tf.reduce_sum(c, axis=0)
        predicted_set = tf.reduce_sum(c, axis=1)

        union = ground_truth_set + predicted_set - intersection
        iou = intersection / (union + 1e-8)
        mean_iou = tf.reduce_sum(iou) / uni_size
        
        return mean_iou 

    def freq_weight_IOU(self, y_true, y_pred, num_clas = 5):

        flatten_pred = tf.reshape(y_pred, [-1])
        flatten_true = tf.reshape(y_true, [-1])

        con = tf.concat([flatten_true, flatten_pred], axis=-1) 
        uni_size = tf.cast(tf.size(tf.unique(con)[0]), dtype=tf.float32)

        c = tf.cast(tf.confusion_matrix(flatten_true, flatten_pred, num_classes=num_clas), dtype=tf.float32)
        intersection = tf.diag_part(c)
        ground_truth_set = tf.reduce_sum(c, axis=0)
        predicted_set = tf.reduce_sum(c, axis=1)

        union = ground_truth_set + predicted_set - intersection

        keys = tf.range(num_clas)
        sorted_inter = tf.gather(intersection, tf.nn.top_k(keys, k=num_clas).indices)
        sorted_union = tf.gather(union, tf.nn.top_k(keys, k=num_clas).indices)

        mask = tf.concat([tf.cast(flatten_true, dtype=tf.int32), keys], axis=-1)
        mask_classes, _, mask_class_weights = tf.unique_with_counts(mask)
        reordered_mask_class_weights = tf.gather(mask_class_weights, tf.nn.top_k(mask_classes, k=num_clas).indices)
        bool_mask = tf.not_equal(reordered_mask_class_weights, [1])

        classes, _, class_weights = tf.unique_with_counts(flatten_true)
        true_class_size = tf.cast(tf.size(tf.unique(flatten_true)[0]), dtype=tf.int32)
        reordered_class_weights = tf.cast(tf.gather(class_weights, tf.nn.top_k(classes, k=true_class_size).indices), dtype=tf.float32)  
   
        total_pixels = tf.cast(tf.shape(flatten_true)[0], dtype=tf.float32)
        non_zero_inter = tf.boolean_mask(sorted_inter, bool_mask)
        numerator = tf.multiply(reordered_class_weights,non_zero_inter)
        denominator = tf.boolean_mask(sorted_union, bool_mask)
     
        weighted_iou = numerator / denominator
        freq_weighted_iou =  tf.reduce_sum(weighted_iou) / total_pixels
        
        return freq_weighted_iou 

    def get_checkpoint_filename(self, model_name, run_name):
        """ 
        Return filename for a checkpoint file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full checkpoint filepath
        """
        pth = os.path.join(self.checkpoint_path, model_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return os.path.join(pth, run_name + '.ckpt')

    def get_summaries_path(self, model_name, run_name):
        """ 
        Return filename for a summaries file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full summaries filepath
        """
        pth = os.path.join(self.summary_path, model_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return os.path.join(pth, run_name)

    def get_results_path(self, model_name, run_name):
        """ 
        Return filename for a results file. Ensure path exists
        Input: model_name - Name of the model
               run_name - Timestap of the training 
        Output: Full results filepath
        """
        pth = os.path.join(self.results_path, model_name, run_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return pth
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
       
        self.train_fn =  os.path.join(self.main_dir, 'tfrecords/train.tfrecords')
        self.val_fn =  os.path.join(self.main_dir, 'tfrecords/val.tfrecords')

        self.pretrain_train_fn =  os.path.join(self.main_dir, 'tfrecords/pretraining_gleason_train.tfrecords')
        self.pretrain_val_fn =  os.path.join(self.main_dir, 'tfrecords/pretraining_gleason_val.tfrecords')

        self.test_fn =  os.path.join(self.main_dir, 'tfrecords/test.tfrecords')

        self.exp_fn = os.path.join(self.main_dir, 'tfrecords/exp.tfrecords')
        
        self.optimizer = "adam"
        self.momentum = 0.9 # if optimizer is nestrov

        self.initial_learning_rate = 0.01
        self.decay_learning_rate = True
        self.decay_steps = 10000 # number of steps before decaying the learning rate
        self.learning_rate_decay_factor = 0.1 
        
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
                                        'grayscale':True
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


    def weighted_cross_entropy(self, onehot_labels, flatten_train_logits, trinary, class_weights):

        class_weights = np.load(class_weights).astype(np.float32)
        if trinary == 1:
            class_weights = np.array([class_weights[0], class_weights[1], class_weights[2]*class_weights[3]*class_weights[4]])
        tf_class_weights = tf.constant(class_weights)
        weight_map = tf.multiply(onehot_labels, tf_class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = onehot_labels, logits = flatten_train_logits)

        weighted_batch_loss = tf.multiply(batch_loss, weight_map)

        return weighted_batch_loss


    # def weighted_cross_entropy(self, onehot_labels, flatten_train_logits, trinary, class_weights):

    #     ground_truth = onehot_labels
    #     ones_count = tf.cast(tf.equal(scores, 1), dtype=tf.float32)
    #     twos_count = tf.cast(tf.equal(scores, 2), dtype=tf.float32)
    #     threes_count = tf.cast(tf.equal(scores, 3), dtype=tf.float32)
    #     fours_count = tf.cast(tf.equal(scores, 4), dtype=tf.float32)


    #     class_weights = np.load(class_weights).astype(np.float32)
    #     if trinary == 1:
    #         class_weights = np.array([class_weights[0], class_weights[1], class_weights[2]*class_weights[3]*class_weights[4]])
    #     else:
    #         tf.equal()
    #     tf_class_weights = tf.constant(class_weights)
    #     weight_map = tf.multiply(onehot_labels, tf_class_weights)
    #     weight_map = tf.reduce_sum(weight_map, axis=1)

    #     batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = onehot_labels, logits = flatten_train_logits)

    #     weighted_batch_loss = tf.multiply(batch_loss, weight_map)

    #     return weighted_batch_loss

    def focal_loss(self, onehot_labels, logits, trinary, class_weights, gamma=1.5, paper_alpha_t=False, name=None, scope=None):
      
        predictions = tf.nn.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1.0), predictions, 1.0-predictions)
        # add small value to avoid 0
        epsilon = 1e-8

        class_weights = np.load(class_weights).astype(np.float32)
        if trinary == 1:
            class_weights = np.array([class_weights[0], class_weights[1], class_weights[2]*class_weights[3]*class_weights[4]])
        alpha_t = tf.constant(class_weights)
        if paper_alpha_t:
            alpha_t = 0.25
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1.0-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1.0 - predictions_pt, gamma) * tf.log(predictions_pt+epsilon), axis=1)
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
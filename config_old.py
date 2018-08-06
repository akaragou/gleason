from __future__ import division
import os
import tensorflow as tf
import numpy as np

class Config():
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
        self.initial_learning_rate = 3e-04
        self.momentum = 0.9 # if optimizer is nestrov
        self.train_batch_size = 2
        self.val_batch_size = 2
        self.num_batches_to_validate_over = 50 # number of batches to validate over 32*100 = 3200
        self.validate_every_num_steps = 100 # perform a validation step
        self.num_train_epochs = 10000
        self.output_shape = 5 # output shape of the model if 2 we have binary classification 
        self.input_image_size = [512, 512, 3] # size of the input tf record image
        self.batch_norm = True # needs to be applied to both training and validation graph 

        self.train_augmentations_dic = {
                                        'rand_flip_left_right':True,
                                        'rand_flip_top_bottom':True,
                                        'rand_rotate':True,
                                        'warp':True
                                       }

        self.val_augmentations_dic = {
                                      'rand_flip_left_right':False,
                                      'rand_flip_top_bottom':False,
                                      'rand_rotate':False,
                                      'warp':False
                                     }

    def rccn_loss():
        pass
    def dice_loss():
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


    def IOU(self, y_true, y_pred, num_classes = 5):

        flatten_pred = tf.reshape(y_pred, [-1])
        flatten_true = tf.reshape(y_true, [-1])

        con = tf.concat([flatten_true, flatten_pred], axis=-1) 
        uni = tf.unique(con)
        uni_size =tf.cast(tf.size(uni[0]), dtype=tf.float32)

        c =  tf.cast(tf.confusion_matrix(flatten_true, flatten_pred, num_classes = 5), dtype=tf.float32)
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

import os
import tensorflow as tf

class GleasonConfig():
    def __init__(self, **kwargs):

        # directories for storing tfrecords, checkpoints etc.
        self.main_dir = '/media/data_cifs/andreas/pathology/gleason_training_patches/'
        self.checkpoint_path = os.path.join(self.main_dir, 'checkpoints')
        self.summary_path = os.path.join(self.main_dir, 'summaries')
        self.results_path = os.path.join(self.main_dir, 'results')
        self.model_path = '/media/data_cifs/andreas/model_weights/resnet_v2_50.ckpt'

        self.train_fn =  os.path.join(self.main_dir, 'tfrecords/train.tfrecords')
        self.val_fn =  os.path.join(self.main_dir, 'tfrecords/val.tfrecords')
        self.test_fn =  os.path.join(self.main_dir, 'tfrecords/test.tfrecords')

        self.restore = False
        self.optimizer = "nestrov"
        self.l2_reg = 0.001
        self.initial_learning_rate = 0.001
        self.momentum = 0.9 # if optimizer is nestrov
        self.decay_steps = 5000 # number of steps before decaying the learning rate
        self.learning_rate_decay_factor = 0.1
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.num_batches_to_validate_over = 131 # number of batches to validate over 32*100 = 3200
        self.validate_every_num_steps = 100 # perform a validation step
        self.num_train_epochs = 10000
        self.input_image_size = [256, 256, 3] # size of the input tf record image
        self.model_image_size = [224, 224, 3] # image dimesions that the model takes in
        self.preprocess_batchnorm = True

        # various options for altering input images during training and validation
        self.train_augmentations_dic = {
                                        'rand_flip_left_right':True,
                                        'rand_flip_top_bottom':True,
                                        'rand_crop': True,
                                        'rand_rotate':True,
                                        'warp':True,
                                        'grayscale':True
                                        }

        self.val_augmentations_dic = {
                                    'rand_flip_left_right':False,
                                    'rand_flip_top_bottom':False,
                                    'rand_crop': False,
                                    'rand_rotate':False,
                                    'warp':False,
                                    'grayscale':True
                                     }

        # saliency map config
        self.target_gradients = True
        self.plot = True
        self.save_masked_image = False
        self.mu = 0
        self.sd = 0.16
        self.num_iter = 50
        self.saliency_map_batch_size = 1
        self.output_masked_img_dir =  '/media/data_cifs/andreas/pathology/mask_slide_crops'
    
    
    def class_accuracy(self, logits, labels):
        """
        Computes class accuracy during training and validation.
        Input: logits - Mini-batch logits Tensor 
               labels - Mini-batch class labels Tensor
        Output: Class accuracy for that mini-batch
        """
        return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, 1), tf.cast(labels, dtype=tf.int64))))


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
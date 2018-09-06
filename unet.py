from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 

def conv_residual_block(net, num_features, is_training, is_batch_norm, layer_name):
    """ Unet_V2 residual conv block 
        Inputs: net - input feature map
                num_features - number of features in convolution block
                is_training - boolean whether to train graph or validate/test
                is_batch_norm - boolean whether to have batchnorm activated or not
                layer_name - scope name for layer

        Output: net - a feature map 
    """
    with tf.variable_scope(layer_name):
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_2' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm2')
        shortcut = tf.nn.relu(net)
        net = shortcut
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_3' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm3')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_features, [3,3], activation_fn=None,  normalizer_fn=None, scope='conv%d_4' % int(layer_name[-1]))
        if is_batch_norm:
            net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                    epsilon=1e-5, center=True, scale=True,scope='batch_norm4')
        net = tf.nn.relu(net + shortcut)
        return net

def conv_bn_relu(net, num_features, is_training, is_batch_norm, scope_name):
    net = slim.conv2d(net, num_features, [3,3], activation_fn=None, normalizer_fn=None, scope=scope_name)
    if is_batch_norm:
        net = slim.batch_norm(net, is_training=is_training, decay=0.997, 
                epsilon=1e-5, center=True, scale=True,scope='batch_norm_' + scope_name.split('/')[-1] )
    net = tf.nn.relu(net)
    return net


def unet_arg_scope(weight_decay=0.0005):
  """Defines the Unet arg scope.
    Input: weight_decay - The l2 regularization coefficient
    Output: arg_scope - argument scope of model
    """
  with slim.arg_scope([slim.conv2d],
                      padding='SAME',
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()) as arg_sc:
    return arg_sc

##########################################
# Unet: https://arxiv.org/abs/1505.04597 #
##########################################

def Unet(inputs,
         is_training = True,
         is_batch_norm = True,
         num_classes = 5,
         scope='unet'):


    with tf.variable_scope(scope, 'unet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
        with slim.arg_scope([slim.conv2d,slim.conv2d_transpose, slim.max_pool2d],
                            outputs_collections=end_points_collection):

            ######################
            # downsampling  path #
            ######################
            conv1_1 = conv_bn_relu(inputs, 64, is_training, is_batch_norm, scope_name='conv1/conv1_1')
            conv1_2 = conv_bn_relu(conv1_1, 64, is_training, is_batch_norm, scope_name='conv1/conv1_2')
            pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

            conv2_1 = conv_bn_relu(pool1, 128, is_training, is_batch_norm, scope_name='conv2/conv2_1')
            conv2_2 = conv_bn_relu(conv2_1, 128, is_training, is_batch_norm, scope_name='conv2/conv2_2')
            pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

            conv3_1 = conv_bn_relu(pool2, 256, is_training, is_batch_norm, scope_name='conv3/conv3_1')
            conv3_2 = conv_bn_relu(conv3_1, 256, is_training, is_batch_norm, scope_name='conv3/conv3_2')
            pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

            conv4_1 = conv_bn_relu(pool3, 512, is_training, is_batch_norm, scope_name='conv4/conv4_1')
            conv4_2 = conv_bn_relu(conv4_1, 512, is_training, is_batch_norm, scope_name='conv4/conv4_2')
            pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')

            ##############
            # bottleneck #
            ##############
            conv5_1 = conv_bn_relu(pool4, 1024, is_training, is_batch_norm, scope_name='conv5/conv5_1')
            conv5_2 = conv_bn_relu(conv5_1, 1024, is_training, is_batch_norm, scope_name='conv5/conv5_2')

            ###################
            # upsampling path #
            ###################
            conv6_1 = slim.conv2d_transpose(conv5_2, 512, [2,2], stride=2, scope='conv6/transpose_conv6_1')
            merge_1 = tf.concat([conv6_1, conv4_2], axis=-1, name='merge1') 
            conv6_2 = conv_bn_relu(merge_1, 512, is_training, is_batch_norm, scope_name='conv6/conv6_2')
            conv6_3 = conv_bn_relu(conv6_2, 512, is_training, is_batch_norm, scope_name='conv6/conv6_3')

            conv7_1 = slim.conv2d_transpose(conv6_3, 256, [2,2], stride=2, scope = 'conv7/transpose_conv7_1')
            merge_2 = tf.concat([conv7_1, conv3_2], axis=-1, name='merge2')
            conv7_2 = conv_bn_relu(merge_2, 256, is_training, is_batch_norm, scope_name='conv7/conv7_2')
            conv7_3 = conv_bn_relu(conv7_2, 256, is_training, is_batch_norm, scope_name='conv7/conv7_3')

            conv8_1 = slim.conv2d_transpose(conv7_3, 128, [2,2], stride=2, scope = 'conv8/transpose_conv8_1')
            merge_3 = tf.concat([conv8_1, conv2_2], axis=-1, name='merge3') 
            conv8_2 = conv_bn_relu(merge_3, 128, is_training, is_batch_norm, scope_name='conv8/conv8_2')
            conv8_3 = conv_bn_relu(conv8_2, 128, is_training, is_batch_norm, scope_name='conv8/conv8_3')

            conv9_1 = slim.conv2d_transpose(conv8_3, 64, [2,2], stride=2, scope = 'conv9/transpose_conv9_1')
            merge_4 = tf.concat([conv9_1, conv1_2], axis=-1, name='merge4') 
            conv9_2 = conv_bn_relu(merge_4, 64, is_training, is_batch_norm, scope_name='conv9/conv9_2')
            conv9_3 = conv_bn_relu(conv9_2, 64, is_training, is_batch_norm, scope_name='conv9/conv9_3')

            ###############
            # outpput map #
            ###############
            output_map = slim.conv2d(conv9_3, num_classes, [1, 1], 
                                    activation_fn=None, normalizer_fn=None, 
                                    scope='output_layer')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
           
            return output_map, end_points


##############################################
# FusionNet: https://arxiv.org/abs/1612.05360 #
###############################################

def ResidualUnet(inputs,
         num_classes = 5,
         is_training = True,
         is_batch_norm = False,
         scope='fusionNet'):

    with tf.variable_scope(scope, 'fusionNet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            outputs_collections=end_points_collection):

            ######################
            # downsampling  path #
            ######################
            conv1_1 = conv_bn_relu(inputs, 64, is_training, is_batch_norm, 'conv1/conv1_1')
            conv1 = conv_residual_block(conv1_1, 64, is_training, is_batch_norm, 'conv1')
            conv1_5 = conv_bn_relu(conv1, 64, is_training, is_batch_norm, 'conv1/conv1_5')
            pool1 = slim.max_pool2d(conv1_5, [2, 2], scope='pool1')

            conv2_1 = conv_bn_relu(pool1, 128, is_training, is_batch_norm, 'conv2/conv2_1')
            conv2 = conv_residual_block(conv2_1, 128, is_training, is_batch_norm, 'conv2')
            conv2_5 = conv_bn_relu(conv2, 128, is_training, is_batch_norm, 'conv2/conv2_5')
            pool2 = slim.max_pool2d(conv2_5, [2, 2], scope='pool2')

            conv3_1 = conv_bn_relu(pool2, 256, is_training, is_batch_norm, 'conv3/conv3_1')
            conv3 = conv_residual_block(conv3_1, 256, is_training, is_batch_norm, 'conv3')
            conv3_5 = conv_bn_relu(conv3, 256, is_training, is_batch_norm, 'conv3/conv3_5')
            pool3 = slim.max_pool2d(conv3_5, [2, 2], scope='pool3')

            conv4_1 = conv_bn_relu(pool3, 512, is_training, is_batch_norm, 'conv4/conv4_1')
            conv4 = conv_residual_block(conv4_1, 512, is_training, is_batch_norm, 'conv4')
            conv4_5 = conv_bn_relu(conv4, 512, is_training, is_batch_norm, 'conv4/conv4_5')
            pool4 = slim.max_pool2d(conv4_5, [2, 2], scope='pool4')


            ##############
            # bottleneck #
            ##############
            conv5_1 = conv_bn_relu(pool4, 1024, is_training, is_batch_norm, 'conv5/conv5_1')
            conv5 = conv_residual_block(conv5_1, 1024, is_training, is_batch_norm, 'conv5')
            conv5_5 = conv_bn_relu(conv5, 1024, is_training, is_batch_norm, 'conv5/conv5_5')

            ###################
            # upsampling path #
            ###################
            conv6_up = slim.conv2d_transpose(conv5_5, 512, [2,2], activation_fn=None,  normalizer_fn=None, stride=2, scope='conv6/transpose_conv6')
            conv6_up += conv4_5
            conv6_up = tf.nn.relu(conv6_up)
            conv6_1 = conv_bn_relu(conv6_up, 512, is_training, is_batch_norm, 'conv6/conv6_1')
            conv6 = conv_residual_block(conv6_1, 512, is_training, is_batch_norm, 'conv6')
            conv6_5 = conv_bn_relu(conv6, 512, is_training, is_batch_norm, 'conv6/conv6_5')
     
            conv7_up = slim.conv2d_transpose(conv6_5, 256, [2,2], activation_fn=None,  normalizer_fn=None, stride=2, scope='conv7/transpose_conv7')
            conv7_up += conv3_5
            conv7_up = tf.nn.relu(conv7_up)
            conv7_1 = conv_bn_relu(conv7_up, 256, is_training, is_batch_norm, 'conv7/conv7_1')
            conv7 = conv_residual_block(conv7_1, 256, is_training, is_batch_norm, 'conv7')
            conv7_5 = conv_bn_relu(conv7, 256, is_training, is_batch_norm, 'conv7/conv7_5')

            conv8_up = slim.conv2d_transpose(conv7_5, 128, [2,2], activation_fn=None,  normalizer_fn=None,  stride=2, scope='conv8/transpose_conv8')
            conv8_up += conv2_5
            conv8_up = tf.nn.relu(conv8_up)
            conv8_1 = conv_bn_relu(conv8_up, 128, is_training, is_batch_norm, 'conv8/conv8_1')
            conv8 = conv_residual_block(conv8_1, 128, is_training, is_batch_norm, 'conv8')
            conv8_5 = conv_bn_relu(conv8, 128, is_training, is_batch_norm, 'conv8/conv8_5')

            conv9_up = slim.conv2d_transpose(conv8_5, 64, [2,2], activation_fn=None,  normalizer_fn=None, stride=2, scope='conv9/transpose_conv9')
            conv9_up += conv1_5
            conv9_up = tf.nn.relu(conv9_up)
            conv9_1 = conv_bn_relu(conv9_up, 64, is_training, is_batch_norm, 'conv9/conv9_1')
            conv9 = conv_residual_block(conv9_1, 64, is_training, is_batch_norm, 'conv9')
            conv9_5 = conv_bn_relu(conv9, 64, is_training, is_batch_norm, 'conv9/conv9_5')

            ###############
            # outpput map #
            ###############
            output_map = slim.conv2d(conv9_5, num_classes, [3, 3], 
                                    activation_fn=None, normalizer_fn=None, 
                                    scope='output_layer')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return output_map, end_points













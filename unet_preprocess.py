from __future__ import division 
import tensorflow as tf 
slim = tf.contrib.slim 

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

def unet(inputs,
         is_training = True,
         is_batch_norm = True,
         num_channels = 3,
         scope='unet'):


    with tf.variable_scope(scope, 'unet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, and max_pool2d.
        with slim.arg_scope([slim.conv2d,slim.conv2d_transpose, slim.max_pool2d],
                            outputs_collections=end_points_collection):

            ######################
            # downsampling  path #
            ######################
            conv1_1 = conv_bn_relu(inputs, 32, is_training, is_batch_norm, scope_name='conv1/conv1_1')
            conv1_2 = conv_bn_relu(conv1_1, 32, is_training, is_batch_norm, scope_name='conv1/conv1_2')
            pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

            conv2_1 = conv_bn_relu(pool1, 64, is_training, is_batch_norm, scope_name='conv2/conv2_1')
            conv2_2 = conv_bn_relu(conv2_1, 64, is_training, is_batch_norm, scope_name='conv2/conv2_2')
            pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

            conv3_1 = conv_bn_relu(pool2, 128, is_training, is_batch_norm, scope_name='conv3/conv3_1')
            conv3_2 = conv_bn_relu(conv3_1, 128, is_training, is_batch_norm, scope_name='conv3/conv3_2')
            pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

            conv4_1 = conv_bn_relu(pool3, 256, is_training, is_batch_norm, scope_name='conv4/conv4_1')
            conv4_2 = conv_bn_relu(conv4_1, 256, is_training, is_batch_norm, scope_name='conv4/conv4_2')
            pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')

            ##############
            # bottleneck #
            ##############
            conv5_1 = conv_bn_relu(pool4, 512, is_training, is_batch_norm, scope_name='conv5/conv5_1')
            conv5_2 = conv_bn_relu(conv5_1, 512, is_training, is_batch_norm, scope_name='conv5/conv5_2')

            ###################
            # upsampling path #
            ###################
            conv6_1 = slim.conv2d_transpose(conv5_2, 512, [2,2], activation_fn=None, stride=2, scope='conv6/transpose_conv6_1')
            merge_1 = tf.concat([conv6_1, conv4_2], axis=-1, name='merge1') 
            conv6_2 = conv_bn_relu(merge_1, 256, is_training, is_batch_norm, scope_name='conv6/conv6_2')
            conv6_3 = conv_bn_relu(conv6_2, 256, is_training, is_batch_norm, scope_name='conv6/conv6_3')

            conv7_1 = slim.conv2d_transpose(conv6_3, 256, [2,2], activation_fn=None, stride=2, scope = 'conv7/transpose_conv7_1')
            merge_2 = tf.concat([conv7_1, conv3_2], axis=-1, name='merge2')
            conv7_2 = conv_bn_relu(merge_2, 128, is_training, is_batch_norm, scope_name='conv7/conv7_2')
            conv7_3 = conv_bn_relu(conv7_2, 128, is_training, is_batch_norm, scope_name='conv7/conv7_3')

            conv8_1 = slim.conv2d_transpose(conv7_3, 128, [2,2], activation_fn=None, stride=2, scope = 'conv8/transpose_conv8_1')
            merge_3 = tf.concat([conv8_1, conv2_2], axis=-1, name='merge3') 
            conv8_2 = conv_bn_relu(merge_3, 64, is_training, is_batch_norm, scope_name='conv8/conv8_2')
            conv8_3 = conv_bn_relu(conv8_2, 64, is_training, is_batch_norm, scope_name='conv8/conv8_3')

            conv9_1 = slim.conv2d_transpose(conv8_3, 64, [2,2], activation_fn=None, stride=2, scope = 'conv9/transpose_conv9_1')
            merge_4 = tf.concat([conv9_1, conv1_2], axis=-1, name='merge4') 
            conv9_2 = conv_bn_relu(merge_4, 32, is_training, is_batch_norm, scope_name='conv9/conv9_2')
            conv9_3 = conv_bn_relu(conv9_2, 32, is_training, is_batch_norm, scope_name='conv9/conv9_3')

            ###############
            # outpput map #
            ###############
            output_map = slim.conv2d(conv9_3, num_channels, [3, 3], 
                                    activation_fn=None, normalizer_fn=None, 
                                    scope='output_layer')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
           
            return output_map, end_points


# def unet(inputs,
#          is_training = True,
#          is_batch_norm = True,
#          num_channels = 3,
#          scope='unet'):


#     with tf.variable_scope(scope, 'unet', [inputs]) as sc:
#         end_points_collection = sc.name + '_end_points'
#         # Collect outputs for conv2d, and max_pool2d.
#         with slim.arg_scope([slim.conv2d,slim.conv2d_transpose, slim.max_pool2d],
#                             outputs_collections=end_points_collection):

#             ######################
#             # downsampling  path #
#             ######################
#             conv1_1 = conv_bn_relu(inputs, 16, is_training, is_batch_norm, scope_name='conv1/conv1_1')
#             conv1_2 = conv_bn_relu(conv1_1, 16, is_training, is_batch_norm, scope_name='conv1/conv1_2')
#             pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

#             conv2_1 = conv_bn_relu(pool1, 32, is_training, is_batch_norm, scope_name='conv2/conv2_1')
#             conv2_2 = conv_bn_relu(conv2_1, 32, is_training, is_batch_norm, scope_name='conv2/conv2_2')
#             pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

#             conv3_1 = conv_bn_relu(pool2, 64, is_training, is_batch_norm, scope_name='conv3/conv3_1')
#             conv3_2 = conv_bn_relu(conv3_1, 64, is_training, is_batch_norm, scope_name='conv3/conv3_2')
#             pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

#             conv4_1 = conv_bn_relu(pool3, 128, is_training, is_batch_norm, scope_name='conv4/conv4_1')
#             conv4_2 = conv_bn_relu(conv4_1, 128, is_training, is_batch_norm, scope_name='conv4/conv4_2')
#             pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')

#             ##############
#             # bottleneck #
#             ##############
#             conv5_1 = conv_bn_relu(pool4, 256, is_training, is_batch_norm, scope_name='conv5/conv5_1')
#             conv5_2 = conv_bn_relu(conv5_1, 256, is_training, is_batch_norm, scope_name='conv5/conv5_2')

#             ###################
#             # upsampling path #
#             ###################
#             conv6_1 = slim.conv2d_transpose(conv5_2, 256, [2,2], activation_fn=None, stride=2, scope='conv6/transpose_conv6_1')
#             merge_1 = tf.concat([conv6_1, conv4_2], axis=-1, name='merge1') 
#             conv6_2 = conv_bn_relu(merge_1, 128, is_training, is_batch_norm, scope_name='conv6/conv6_2')
#             conv6_3 = conv_bn_relu(conv6_2, 128, is_training, is_batch_norm, scope_name='conv6/conv6_3')

#             conv7_1 = slim.conv2d_transpose(conv6_3, 128, [2,2], activation_fn=None, stride=2, scope = 'conv7/transpose_conv7_1')
#             merge_2 = tf.concat([conv7_1, conv3_2], axis=-1, name='merge2')
#             conv7_2 = conv_bn_relu(merge_2, 64, is_training, is_batch_norm, scope_name='conv7/conv7_2')
#             conv7_3 = conv_bn_relu(conv7_2, 64, is_training, is_batch_norm, scope_name='conv7/conv7_3')

#             conv8_1 = slim.conv2d_transpose(conv7_3, 64, [2,2], activation_fn=None, stride=2, scope = 'conv8/transpose_conv8_1')
#             merge_3 = tf.concat([conv8_1, conv2_2], axis=-1, name='merge3') 
#             conv8_2 = conv_bn_relu(merge_3, 32, is_training, is_batch_norm, scope_name='conv8/conv8_2')
#             conv8_3 = conv_bn_relu(conv8_2, 32, is_training, is_batch_norm, scope_name='conv8/conv8_3')

#             conv9_1 = slim.conv2d_transpose(conv8_3, 32, [2,2], activation_fn=None, stride=2, scope = 'conv9/transpose_conv9_1')
#             merge_4 = tf.concat([conv9_1, conv1_2], axis=-1, name='merge4') 
#             conv9_2 = conv_bn_relu(merge_4, 16, is_training, is_batch_norm, scope_name='conv9/conv9_2')
#             conv9_3 = conv_bn_relu(conv9_2, 16, is_training, is_batch_norm, scope_name='conv9/conv9_3')

#             ###############
#             # outpput map #
#             ###############
#             output_map = slim.conv2d(conv9_3, num_channels, [3, 3], 
#                                     activation_fn=None, normalizer_fn=None, 
#                                     scope='output_layer')

#             # Convert end_points_collection into a end_point dict.
#             end_points = slim.utils.convert_collection_to_dict(end_points_collection)
           
#             return output_map, end_points














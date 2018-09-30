# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir',
  'datasets/cityscapes/exp/train_on_trainval_set/modify/',
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', True,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [769, 769],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', 'datasets/cityscapes/exp/train_on_trainval_set/train_vis/model.ckpt-9699',
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', False,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'cityscapes',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', 'datasets/cityscapes/tfrecord', 'Where the dataset reside.')

def _build_deeplab(inputs_queue, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    inputs_queue: A prefetch queue for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.
  """
  samples = inputs_queue.dequeue()

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(
      samples[common.LABEL], name=common.LABEL)

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=FLAGS.train_crop_size,
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)
  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      image_pyramid=FLAGS.image_pyramid,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

  # Add name to graph node so we can add to summary.
  output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
  output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
      output_type_dict[model.MERGED_LOGITS_SCOPE],
      name=common.OUTPUT_TYPE)

  if True:
    graph = tf.get_default_graph()
    end_point_name = [
      "xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3_pointwise/Relu:0",
      "decoder/decoder_conv0_pointwise/Relu:0",
      "decoder/decoder_conv1_pointwise/Relu:0"]
    end_point_alias = [
      "net1/",
      "net2/",
      "net3/"
    ]
    end_point = []
    for n in end_point_name:
      end_point.append(graph.get_tensor_by_name(n))

    draw_images = []
    for x, a in zip(end_point, end_point_alias):
      with tf.variable_scope("vis", reuse=tf.AUTO_REUSE):
        draw_feature = tf.layers.conv2d(x, 512, 1, name=a+"compress_feat1", activation=tf.nn.relu)
        draw_feature = tf.layers.conv2d(draw_feature, 256, 1, name=a+"compress_feat2", activation=tf.nn.relu)
        image = tf.layers.conv2d(draw_feature, 3, 3, padding='SAME', name="draw", activation=tf.nn.tanh)
        draw_images.append(image)

    # Add visualization
    visualize_images = []
    target_size = samples[common.IMAGE].get_shape().as_list()[1:3]
    for vis_im, name in zip(draw_images, end_point_alias):
      im_resize = tf.image.resize_bilinear(vis_im, target_size, align_corners=True)
      visualize_images.append(im_resize)
      tf.summary.image('reconstruction/' + name, im_resize)
      # im_tar = tf.Print(samples[common.IMAGE], [tf.reduce_max(samples[common.IMAGE]), tf.reduce_min(samples[common.IMAGE])])
      im_tar = samples[common.IMAGE] / 127.5 - 1.0
      reconstruction_loss = tf.reduce_mean(tf.abs(im_resize - im_tar))
      reconstruction_loss = tf.identity(reconstruction_loss, name=name+"recons_loss")
      tf.add_to_collection(tf.GraphKeys.LOSSES, reconstruction_loss)

  return outputs_to_scales_to_logits, visualize_images, end_point


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', FLAGS.train_split)

  if True:
    graph = tf.get_default_graph()
    with tf.device(config.inputs_device()):
      samples = input_generator.get(
          dataset,
          FLAGS.train_crop_size,
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=32 * config.num_clones)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      net_output, images, end_points = _build_deeplab(
        inputs_queue,
        {common.OUTPUT_TYPE: dataset.num_classes},
        dataset.ignore_label)
    
    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=session_config)
    tf.train.start_queue_runners(sess=sess)
    train_utils.get_model_init_fn(
            FLAGS.train_logdir,
            FLAGS.tf_initial_checkpoint,
            True,
            None,
            ignore_missing_vars=True)(sess)

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.tf_initial_checkpoint)

    net_input = graph.get_tensor_by_name('image:0')
    net_image = images[-1]
    gen_feature = end_points[0]
    predictions = graph.get_tensor_by_name('semantic:0')
    num_classes = int(predictions.get_shape()[-1])
    segmentation = tf.argmax(predictions, 3)
    groundtruth = graph.get_tensor_by_name('label:0')
    masks = []
    masks_area = []
    for i in range(num_classes):
        masks.append(tf.cast(tf.equal(groundtruth, i), tf.float32))
        masks_area.append(tf.reduce_sum(masks[-1]))

    # id=0: road
    masked_losses = []
    for i in range(num_classes):
        masked_diff = tf.abs(masks[i] * (net_image - net_input))
        masked_losses.append(
            tf.reduce_sum(masked_diff) / (1e-6 + masks_area[i])
            )
        class_grads.extend(
            masked_losses[-1], gen_feature
            ))

    grad = tf.gradients(masked_losses[0], gen_feature)

    lr = 1.0
    new_gen_feature = gen_feature - lr * grad
    new_output = 

    print("=> Test")
    net_output_data, images_data, end_points_data = sess.run([net_output, images, end_points])
    return sess, net_output, images, end_points

if __name__ == '__main__':
  sess, net_output, images, end_points = main(0)
  net_output_data, images_data, end_points_data = sess.run([net_output, images, end_points])
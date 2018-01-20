
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(labels[index])),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())


def main(unused_argv):
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='./Images/violent',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=1,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  """
  import random
  import tensorflow as tf
  from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

  #===============DEFINE YOUR ARGUMENTS==============
  flags = tf.app.flags

  #State your dataset directory
  flags.DEFINE_string('dataset_dir', "Images", 'String: Your dataset directory')

  # The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
  flags.DEFINE_float('validation_size', 0, 'Float: The proportion of examples in the dataset to be used for validation')

  # The number of shards per dataset split.
  flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

  # Seed for repeatability.
  flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

  #Output filename for the naming the TFRecord file
  flags.DEFINE_string('tfrecord_filename', "converted_images", 'String: The output filename to name your TFRecord file')

  FLAGS = flags.FLAGS

  def main():

      #=============CHECKS==============
      #Check if there is a tfrecord_filename entered
      if not FLAGS.tfrecord_filename:
          raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

      #Check if there is a dataset directory entered
      if not FLAGS.dataset_dir:
          raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

      #If the TFRecord files already exist in the directory, then exit without creating the files again
      if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
          print ('Dataset files already exist. Exiting without re-creating them.')
          return None
      #==========END OF CHECKS============

      #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
      photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

      #Refer each of the class name to a specific integer number for predictions later
      class_names_to_ids = dict(zip(class_names, range(len(class_names))))

      #Find the number of validation examples we need
      num_validation = int(FLAGS.validation_size * len(photo_filenames))

      # Divide the training datasets into train and test:
      random.seed(FLAGS.random_seed)
      random.shuffle(photo_filenames)
      training_filenames = photo_filenames[num_validation:]
      validation_filenames = photo_filenames[:num_validation]

      # First, convert the training and validation sets.
      _convert_dataset('train', training_filenames, class_names_to_ids,
                       dataset_dir = FLAGS.dataset_dir,
                       tfrecord_filename = FLAGS.tfrecord_filename,
                       _NUM_SHARDS = FLAGS.num_shards)
      _convert_dataset('validation', validation_filenames, class_names_to_ids,
                       dataset_dir = FLAGS.dataset_dir,
                       tfrecord_filename = FLAGS.tfrecord_filename,
                       _NUM_SHARDS = FLAGS.num_shards)

      # Finally, write the labels file:
      labels_to_class_names = dict(zip(range(len(class_names)), class_names))
      write_label_file(labels_to_class_names, FLAGS.dataset_dir)

      print ('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

  if __name__ == "__main__":
      main()
  """
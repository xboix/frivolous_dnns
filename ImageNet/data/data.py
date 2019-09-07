import tensorflow as tf
import os
import data.imagenet_preprocessing as imagenet_preprocessing

_NUM_TRAIN_FILES = 1024
_CYCLE_LENGTH = 10
_NUM_THREADS = 5000
_BUFFER_LENGTH = 1024
_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3

class ImagenetDataset:

    def __init__(self, opt):
        self.opt = opt
        self.set_name = 'train'
        self.num_total_images = 0

    def get_filenames(self, set_name, data_dir):
        """Return filenames for dataset."""
        if set_name == 'train':
            return [
                os.path.join(data_dir, 'train-%05d-of-01024' % i)
                for i in range(_NUM_TRAIN_FILES)]
        else:
            return [
                os.path.join(data_dir, 'validation-%05d-of-00128' % i)
                for i in range(128)]




    def create_dataset(self, set_name='train', repeat=False):

        def _parse_function(example_serialized):
            """Parses an Example proto containing a training example of an image.
            The output of the build_image_data.py image preprocessing script is a dataset
            containing serialized Example protocol buffers. Each Example proto contains
            the following fields (values are included as examples):
              image/height: 462
              image/width: 581
              image/colorspace: 'RGB'
              image/channels: 3
              image/class/label: 615
              image/class/synset: 'n03623198'
              image/class/text: 'knee pad'
              image/object/bbox/xmin: 0.1
              image/object/bbox/xmax: 0.9
              image/object/bbox/ymin: 0.2
              image/object/bbox/ymax: 0.6
              image/object/bbox/label: 615
              image/format: 'JPEG'
              image/filename: 'ILSVRC2012_val_00041207.JPEG'
              image/encoded: <JPEG encoded string>
            Args:
              example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.
            Returns:
              image_buffer: Tensor tf.string containing the contents of a JPEG file.
              label: Tensor tf.int32 containing the label.
              bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged as
                [ymin, xmin, ymax, xmax].
            """
            # Dense features in Example proto.
            feature_map = {
                'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                    default_value=''),
                'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                        default_value=-1),
                'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                       default_value=''),
            }
            sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
            # Sparse features in Example proto.
            feature_map.update(
                {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                             'image/object/bbox/ymin',
                                             'image/object/bbox/xmax',
                                             'image/object/bbox/ymax']})

            features = tf.parse_single_example(example_serialized, feature_map)
            label = tf.cast(features['image/class/label'], dtype=tf.int32)

            xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
            ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
            xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
            ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

            # Note that we impose an ordering of (y, x) just to make life difficult.
            bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

            # Force the variable number of bounding boxes into the shape
            # [1, num_boxes, coords].
            bbox = tf.expand_dims(bbox, 0)
            bbox = tf.transpose(bbox, [0, 2, 1])

            if self.opt.dnn.name == 'resnet':
                image = imagenet_preprocessing.preprocess_image(
                    image_buffer=features['image/encoded'],
                    bbox=bbox,
                    label=label,
                    output_height=_DEFAULT_IMAGE_SIZE,
                    output_width=_DEFAULT_IMAGE_SIZE,
                    num_channels=_NUM_CHANNELS,
                    is_training=(self.set_name == 'train'))

            elif self.opt.dnn.name == 'inception':
                image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                # Crop the central region of the image with an area containing 87.5% of
                # the original image.
                image = tf.image.central_crop(image, central_fraction=0.875)

                # Resize the image to the original height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [299, 299],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])

            return image, label

        self.set_name = set_name

        tfrecords_path = self.opt.dataset.log_dir_base

        filenames = self.get_filenames(self.set_name, tfrecords_path)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_parallel_calls=_NUM_THREADS)


        if self.set_name=='train':
            # Shuffle the input files
            dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
            self.num_total_images = self.opt.dataset.num_images_training
        else:
            self.num_total_images = self.opt.dataset.num_images_validation


        #dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        #    tf.data.TFRecordDataset, cycle_length=_CYCLE_LENGTH))

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely.

        #dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #    map_func=_parse_function, batch_size=self.opt.hyper.batch_size, num_parallel_batches=80))

        return  dataset.batch(self.opt.hyper.batch_size)
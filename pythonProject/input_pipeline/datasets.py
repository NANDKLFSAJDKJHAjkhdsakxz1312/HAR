import gin
import logging
import tensorflow as tf
import os

@gin.configurable
def load(name, data_dir):
    if name == "HAPT":
        logging.info(f"Preparing dataset {name}...")

        def parse_tfrecord_fn(example):
            feature_description = {
                'window_data': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
            }
            example = tf.io.parse_single_example(example, feature_description)

            # 解析窗口数据和标签
            window_data = tf.io.parse_tensor(example['window_data'], out_type=tf.float32)
            label = tf.io.parse_tensor(example['label'], out_type=tf.int32)

            return window_data, label

        ds_train = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "Train.tfrecords")]
        ).map(parse_tfrecord_fn)
        ds_val = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "Validation.tfrecords")]
        ).map(parse_tfrecord_fn)
        ds_test = tf.data.TFRecordDataset(
            filenames=[os.path.join(data_dir, "Test.tfrecords")]
        ).map(parse_tfrecord_fn)

        # ...

        return prepare(ds_train, ds_val, ds_test)


@gin.configurable
def prepare(ds_train, ds_val, ds_test,  batch_size, caching):
    # Prepare training dataset

    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(10000 // 10)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size, drop_remainder=True)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test
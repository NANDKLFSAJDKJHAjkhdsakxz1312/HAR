import tensorflow as tf


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
    # 您原先的解析函数
    # ...

# 更换为您的 TFRecord 文件路径
tfrecord_file = r'D:\HAR\pythonProject\Test.tfrecords'

# 创建一个读取 TFRecord 文件的数据集
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

# 遍历数据集的前几个样本
for sample in parsed_dataset.take(500):
    window_data, label = sample
    print("Window data shape:", window_data.shape)
    print("Label:", label.numpy())
    # 这里可以添加更多的打印或检查逻辑

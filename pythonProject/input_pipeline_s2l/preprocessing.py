import os

import gin
import pandas as pd
import re
import tensorflow as tf

@gin.configurable
def preprocessor(file_path,window_size,window_step):
    data_folder = file_path

    # 创建数据集字典
    combined_datasets = {"Test": {}, "Train": {}, "Validation": {}}

    # 遍历文件并加载数据
    for file_name in os.listdir(data_folder):
        match = re.match(r'(acc|gyro)_exp(\d+)_user(\d+)\.txt', file_name)
        if not match:
            continue

        sensor_type = match.group(1)  # 'acc' 或 'gyro'
        experiment_number = int(match.group(2))
        user_number = int(match.group(3))

        data = pd.read_csv(os.path.join(data_folder, file_name), delimiter=" ", header=None)

        if 22 <= user_number <= 27:
            dataset_name = "Test"
        elif 1 <= user_number <= 21:
            dataset_name = "Train"
        elif 28 <= user_number <= 30:
            dataset_name = "Validation"
        else:
            continue

        experiment_key = (experiment_number, user_number)

        if experiment_key not in combined_datasets[dataset_name]:
            combined_datasets[dataset_name][experiment_key] = {"acc": pd.DataFrame(), "gyro": pd.DataFrame()}

        combined_datasets[dataset_name][experiment_key][sensor_type] = data

    # 合并加速度和陀螺仪数据
    for dataset_name, dataset in combined_datasets.items():
        for key, sensors_data in dataset.items():
            acc_data = sensors_data["acc"]
            gyro_data = sensors_data["gyro"]
            if not acc_data.empty and not gyro_data.empty:
                combined_data = pd.concat([acc_data, gyro_data], axis=1)
                dataset[key] = combined_data
            else:
                del dataset[key]  # 删除缺少任一传感器数据的记录



    def normalize_data(data):
        """ 标准化数据 """
        mean = data.mean()
        std = data.std()
        return (data - mean) / std

    # 对每个数据集的每个试验进行标准化
    for dataset_name, dataset in combined_datasets.items():
        for key, value in dataset.items():
            dataset[key] = normalize_data(value)


    def sliding_window(data, size, step, experiment_number, user_number):
        """ 应用滑动窗口技术，并记录实验编号和用户编号 """
        num_windows = ((data.shape[0] - size) // step) + 1
        for i in range(num_windows):
            start = i * step
            end = start + size
            window_data = data[start:end]
            yield window_data, experiment_number, user_number




    # 对每个数据集的每个试验应用滑动窗口
    windowed_datasets = {"Test": [], "Train": [], "Validation": []}
    for dataset_name, dataset in combined_datasets.items():
        for (experiment_number, user_number), data in dataset.items():
            for window_data, exp_num, user_num in sliding_window(data, window_size, window_step, experiment_number, user_number):
                windowed_datasets[dataset_name].append((window_data, exp_num, user_num))



    labels_df = pd.read_csv(os.path.join(data_folder, 'labels.txt'), header=None, delimiter=' ')
    labels_df.columns = ['Experiment', 'User', 'Activity', 'Start', 'End']


    # 将标签值从1-12调整为0-11
    labels_df['Activity'] = labels_df['Activity'] - 1

    def label_for_window(window_start, window_end, labels_df, current_experiment, current_user):
        """ 确定窗口的标签 """
        window_labels = labels_df[(labels_df['Experiment'] == current_experiment) &
                                  (labels_df['User'] == current_user) &
                                  (labels_df['Start'] <= window_end) &
                                  (labels_df['End'] >= window_start)]['Activity']
        if not window_labels.empty:
            return window_labels.mode()[0]
        else:
            return None

    # 为每个窗口分配标签
    labeled_windows = {"Test": [], "Train": [], "Validation": []}

    for dataset_name, dataset_windows in windowed_datasets.items():
        for window_data, experiment_number, user_number in dataset_windows:
            window_start = window_data.index[0]
            window_end = window_data.index[-1]
            label = label_for_window(window_start, window_end, labels_df, experiment_number, user_number)
            if label is not None:
                labeled_windows[dataset_name].append((window_data, label, experiment_number, user_number))







    # 假设 labeled_windows 是一个包含 (window_data, label) 对的列表
    # 例如: labeled_windows['Train'] = [(window_data1, label1), (window_data2, label2), ...]

    def convert_to_tensor_dataset(labeled_windows):
        datasets = {}
        for dataset_name, windows_with_labels in labeled_windows.items():
            # 将窗口数据和标签转换为张量
            window_data_tensors = [tf.convert_to_tensor(window_data.values, dtype=tf.float32) for window_data, label, _, _ in windows_with_labels]
            labels_tensors = [tf.convert_to_tensor(label, dtype=tf.int32) for window_data, label, _, _ in windows_with_labels]

            # 创建 TensorFlow Dataset
            dataset = tf.data.Dataset.from_tensor_slices((window_data_tensors, labels_tensors))
            datasets[dataset_name] = dataset
        return datasets

    tensor_datasets = convert_to_tensor_dataset(labeled_windows)




    def _bytes_feature(value):
        """返回一个 bytes_list 从一个 string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList 不会从 EagerTensor 中解包字符串
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(window_data, label):
        """
        创建一个 tf.train.Example 消息用于写入到 TFRecord 文件。
        """
        feature = {
            'window_data': _bytes_feature(tf.io.serialize_tensor(window_data)),
            'label': _bytes_feature(tf.io.serialize_tensor(label))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    for dataset_name, dataset in tensor_datasets.items():
        # 相对路径
        tfrecord_file_path = os.path.join('input_pipeline_s2l', f'{dataset_name}.tfrecords')
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            for window_data, label in dataset:
                example = serialize_example(window_data, label)
                writer.write(example)

    return labeled_windows,windowed_datasets

'''import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['font.family'] = 'SimSun'  # 设置字体为宋体或其他包含所需字符的字体

def plot_window_data(window_data, label, experiment_number, user_number):
    plt.figure(figsize=(15, 5))

    # 假设前三列是加速度计数据，后三列是陀螺仪数据
    acc_data = window_data.iloc[:, :3]
    gyro_data = window_data.iloc[:, 3:]

    # 绘制加速度计数据
    plt.subplot(1, 2, 1)
    plt.plot(acc_data)
    plt.title(f"加速度计数据 - 标签: {label} (试验: {experiment_number}, 用户: {user_number})")
    plt.xlabel('时间步')
    plt.ylabel('加速度')
    plt.legend(['X轴', 'Y轴', 'Z轴'])

    # 绘制陀螺仪数据
    plt.subplot(1, 2, 2)
    plt.plot(gyro_data)
    plt.title(f"陀螺仪数据 - 标签: {label} (试验: {experiment_number}, 用户: {user_number})")
    plt.xlabel('时间步')
    plt.ylabel('角速度')
    plt.legend(['X轴', 'Y轴', 'Z轴'])

    plt.tight_layout()
    plt.show()



def visualize_random_windows(labeled_windows, num_samples=3):
    for dataset_name, windows_with_labels in labeled_windows.items():
        print(f"数据集: {dataset_name}")
        sample_windows = random.sample(windows_with_labels, num_samples)  # 随机选择一些窗口

        for window_data, label, experiment_number, user_number in sample_windows:
            plot_window_data(window_data, label, experiment_number, user_number)

# 对每个数据集进行可视化
# visualize_random_windows(labeled_windows)'''


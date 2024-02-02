import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from input_pipeline.preprocessing import preprocessor
from models import lstm_model
import gin
from input_pipeline.datasets import load
import tensorflow as tf
import main
from utils import utils_params

import matplotlib
import matplotlib.pyplot as plt

# 设置 matplotlib 以支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题


run_paths = {'path_model_id': 'D:\\HAR\\experiments\\run_2024-01-31T22-29-46-927781', 'path_logs_train': 'D:\\HAR\\experiments\\run_2024-01-31T22-29-46-927781\\logs\\run.log', 'path_ckpts_train': 'D:\\HAR\\experiments\\run_2024-01-31T22-29-46-927781\\ckpts', 'path_gin': 'D:\\HAR\\experiments\\run_2024-01-31T22-29-46-927781\\config_operative.gin'}

# gin-config
gin.parse_config_files_and_bindings(['configs/config.gin'], [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())


labeled_windows ,windowed_datasets= preprocessor()
_, _, ds_test = load()
batch_size = gin.query_parameter('prepare.batch_size')
model = lstm_model(input_shape=(250, 6), num_classes=12,batch_size=batch_size)

checkpoint_dir = run_paths['path_ckpts_train']

# 查找最新的检查点文件
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model
    )
    ckpt.restore(latest_ckpt)
    print(f"Restored from {latest_ckpt}")
else:
    print("No checkpoint found.")





import matplotlib.pyplot as plt
import pandas as pd

# 假设windowed_datasets是您从上述代码中获得的数据
windowed_datasets = windowed_datasets  # 您的windowed_datasets数据

# 选择特定的用户和试验
selected_user = 27 # 例如，用户1
selected_experiment = 55 # 例如，试验1

# 筛选出特定用户和试验的所有窗口
selected_windows = [window for window, experiment, user in windowed_datasets['Test'] if user == selected_user and experiment == selected_experiment]

# 初始化加速度和角速度的总数据集
total_acc_data = pd.DataFrame()
total_gyro_data = pd.DataFrame()


window_step = gin.query_parameter('preprocessor.window_step')

# 对每个窗口的数据进行合并
for i, window in enumerate(selected_windows):
    # 假设前3列是加速度数据，后3列是角速度数据
    acc_data = window.iloc[:, 0:3]
    gyro_data = window.iloc[:, 3:6]

    # 将数据合并到总数据集，考虑窗口步长
    total_acc_data = pd.concat([total_acc_data, acc_data.iloc[-window_step:]], ignore_index=True)
    total_gyro_data = pd.concat([total_gyro_data, gyro_data.iloc[-window_step:]], ignore_index=True)

# 绘制加速度曲线图
plt.figure(figsize=(10, 4))
plt.plot(total_acc_data)
plt.title('Acceleration Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.legend(['X', 'Y', 'Z'])



# 绘制角速度曲线图
plt.figure(figsize=(10, 4))
plt.plot(total_gyro_data)
plt.title('Gyroscope Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.legend(['X', 'Y', 'Z'])


labeled_windows = labeled_windows  # 您的labeled_windows数据

# 提取所选用户和试验的窗口索引区间和标签
window_intervals_labels = [(window.index[0], window.index[-1], label) for window, label, experiment, user in labeled_windows['Test'] if user == selected_user and experiment == selected_experiment]

activity_labels = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
    6: "STAND_TO_SIT",
    7: "SIT_TO_STAND",
    8: "SIT_TO_LIE",
    9: "LIE_TO_SIT",
    10: "STAND_TO_LIE",
    11: "LIE_TO_STAND"
}

# 为每个活动分配一个颜色
activity_colors = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'cyan',
    4: 'magenta',
    5: 'yellow',
    6: 'orange',
    7: 'purple',
    8: 'brown',
    9: 'pink',
    10: 'gray',
    11: 'olive'
}

# 绘制加速度曲线图
plt.figure(figsize=(10, 4))
for start, end, label in window_intervals_labels:
    plt.axvspan(start, end, color=activity_colors[label], alpha=0.3)
plt.plot(total_acc_data)
plt.title('ground_truth:Acceleration Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.legend(['X', 'Y', 'Z'])

# 添加活动图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=activity_colors[label], label=activity_labels[label]) for label in activity_colors]
plt.legend(handles=legend_elements, loc='upper right')


# 绘制角速度曲线图
plt.figure(figsize=(10, 4))
for start, end, label in window_intervals_labels:
    plt.axvspan(start, end, color=activity_colors[label], alpha=0.3)
plt.plot(total_gyro_data)
plt.title('ground_truth:Gyroscope Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.legend(['X', 'Y', 'Z'])

# 添加活动图例
plt.legend(handles=legend_elements, loc='upper right')
plt.show()

import numpy as np

# 设置和ds_test相同的批次大小
'''batch_size = gin.query_parameter('prepare.batch_size')
ds_selected = ds_selected.batch(batch_size,drop_remainder=True)
# 初始化用于保存结果的列表
predicted_labels_with_indices = []
current_index = 0
for batch in ds_selected:
    # 预测整个批次
    predicted_labels_batch = model.predict(batch)
    predicted_labels = np.argmax(predicted_labels_batch, axis=1)

    for predicted_label in predicted_labels:
        original_start_index = current_index
        original_end_index = original_start_index + window_size

        predicted_labels_with_indices.append((original_start_index, original_end_index, predicted_label))

        current_index += window_step'''

# 设置窗口步长和窗口大小
window_step = gin.query_parameter('preprocessor.window_step')  # 窗口步长
window_size = gin.query_parameter('preprocessor.window_size')  # 窗口大小
batch_size = gin.query_parameter('prepare.batch_size')   # 批次大小

# 将selected_windows中的窗口数据转换为TensorFlow张量
window_data_tensors = [tf.convert_to_tensor(window.values, dtype=tf.float32) for window in selected_windows]

# 创建TensorFlow Dataset
ds_selected = tf.data.Dataset.from_tensor_slices(window_data_tensors)
ds_selected = ds_selected.batch(batch_size,drop_remainder=True)
predicted_labels_with_indices = []
current_index = 0
batch_data_accumulated = []

total_windows_count = 0  # 用于统计总窗口数

current_index = 0

for batch in ds_selected:
    batch_data = batch
    current_batch_size = len(batch_data)

    for i in range(current_batch_size):
        original_start_index = current_index + i * window_step
        original_end_index = original_start_index + window_size

        window_data = batch_data[i].numpy()
        batch_data_accumulated.append(window_data)

        if len(batch_data_accumulated) == batch_size:
            batch_data_np = np.array(batch_data_accumulated)
            predicted_labels_batch = model.predict(batch_data_np)
            predicted_labels = np.argmax(predicted_labels_batch, axis=1)

            for j, predicted_label in enumerate(predicted_labels):
                start_index = original_start_index + j * window_step-(batch_size-1) * window_step
                end_index = start_index + window_size
                print(f"窗口索引: 起始 {start_index}, 结束 {end_index}, 标签 {predicted_label}")
                predicted_labels_with_indices.append((start_index, end_index, predicted_label))

            batch_data_accumulated = []

    current_index += current_batch_size * window_step


# 打印总窗口数和predicted_labels_with_indices的长度
print("总窗口数:", total_windows_count)
print("predicted_labels_with_indices 列表长度:", len(predicted_labels_with_indices))





# 绘制加速度曲线图
plt.figure(figsize=(10, 4))
for start_index, end_index, label in predicted_labels_with_indices:
    plt.axvspan(start_index, end_index, color=activity_colors[label], alpha=0.3)
plt.plot(total_acc_data)
plt.title('prediction:Acceleration Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.legend(['X', 'Y', 'Z'])

# 添加活动图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=activity_colors[label], label=activity_labels[label]) for label in activity_colors]
plt.legend(handles=legend_elements, loc='upper right')


# 绘制角速度曲线图
plt.figure(figsize=(10, 4))
for start_index, end_index, label in predicted_labels_with_indices:
    plt.axvspan(start_index, end_index, color=activity_colors[label], alpha=0.3)
plt.plot(total_gyro_data)
plt.title('prediction:Gyroscope Data for Selected User and Experiment')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.legend(['X', 'Y', 'Z'])

# 添加活动图例
plt.legend(handles=legend_elements, loc='upper right')
plt.show()
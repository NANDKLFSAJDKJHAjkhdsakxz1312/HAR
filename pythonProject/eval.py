import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, class_names):
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate(model, run_paths, ds_test):
    # 检查点路径
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

    # 准备评估指标
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 获取类别数
    num_classes = model.output_shape[-1]

    # 初始化混淆矩阵
    confusion_mtx = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return labels, tf.argmax(predictions, axis=1)

    # 迭代测试数据集
    for test_images, test_labels in ds_test:
        labels, preds = test_step(test_images, test_labels)
        # 更新混淆矩阵
        for i in range(len(labels)):
            confusion_mtx[labels[i]][preds[i]] += 1

    # 打印结果
    print(f"Test Loss: {test_loss.result()}")
    print(f"Test Accuracy: {test_accuracy.result() * 100}")

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_mtx)

    # 类别名称列表，请根据实际情况替换
    class_names = [
        "Class 1",
        "Class 2",
        "Class 3",
        "Class 4",
        "Class 5",
        "Class 6",
        "Class 7",
        "Class 8",
        "Class 9",
        "Class 10",
        "Class 11",
        "Class 12"
    ]

    # 绘制混淆矩阵图像
    plot_confusion_matrix(confusion_mtx, class_names)

    # 返回评估指标和混淆矩阵
    return test_loss.result().numpy(), test_accuracy.result().numpy(), confusion_mtx

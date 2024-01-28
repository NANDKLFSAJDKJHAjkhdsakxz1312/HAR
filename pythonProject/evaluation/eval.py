import tensorflow as tf


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
    print('111')

    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    print('222')




    print('333')

    # 迭代测试数据集
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    # 打印结果
    print(f"Test Loss: {test_loss.result()}")
    print(f"Test Accuracy: {test_accuracy.result() * 100}")


    # 返回评估指标
    return test_loss.result().numpy(), test_accuracy.result().numpy()

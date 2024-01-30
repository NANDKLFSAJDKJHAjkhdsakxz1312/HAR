predicted_labels = model.predict(test_data)
predicted_labels = np.argmax(predicted_labels, axis=1)


def plot_predicted_data(experiment_number, user_number, labeled_windows, predicted_labels, dataset_name='Test'):
    plt.figure(figsize=(15, 5))

    # 获取指定试验和用户的窗口
    windows = [(window_data, label) for window_data, label, exp_num, user_num in labeled_windows[dataset_name] if
               exp_num == experiment_number and user_num == user_number]

    if not windows:
        print("没有找到指定试验和用户的数据")
        return

    for i, (window_data, true_label) in enumerate(windows, 1):
        plt.subplot(len(windows), 1, i)
        plt.plot(window_data.iloc[:, :3])  # 假设前三列是特征数据
        plt.title(f"窗口 {i} - 实际标签: {true_label}, 预测标签: {predicted_labels[i - 1]}")
        plt.xlabel('时间步')
        plt.ylabel('传感器值')

    plt.tight_layout()
    plt.show()


plot_predicted_data(5, 23, labeled_windows, predicted_labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import gin

@gin.configurable
def create_crnn_model(input_shape, num_classes, filter_units=64, lstm_units=64, dense_units=128, dropout_rate=0.5):
    model = Sequential()

    # 添加一维卷积层，用于捕捉局部特征
    model.add(Conv1D(filter_units, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # 添加LSTM层，用于捕捉长期依赖关系
    model.add(LSTM(lstm_units, return_sequences=False))

    # 添加全连接层
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))  # 随机丢弃，防止过拟合

    # 输出层，适用于分类任务
    model.add(Dense(num_classes, activation='softmax'))

    return model


# 使用示例
input_shape = (250, 6)  # 根据您的输入数据形状调整
num_classes = 12  # 根据您的任务中的类别数量调整

model = create_crnn_model(input_shape, num_classes)

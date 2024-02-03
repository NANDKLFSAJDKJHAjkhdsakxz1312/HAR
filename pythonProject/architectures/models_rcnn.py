import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense


# 定义CRNN模型
def create_crnn_model(input_shape, num_classes):
    # 输入层
    input_layer = Input(shape=input_shape)

    # 卷积层
    conv1 = Conv1D(32, kernel_size=3, activation='relu')(input_layer)
    maxpool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(64, kernel_size=3, activation='relu')(maxpool1)
    maxpool2 = MaxPooling1D(pool_size=2)(conv2)

    # 双向LSTM层
    lstm = Bidirectional(LSTM(64, return_sequences=False))(maxpool2)

    # 密集层
    dense = Dense(64, activation='relu')(lstm)

    # 输出层
    output_layer = Dense(num_classes, activation='softmax')(dense)

    # 创建模型
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model




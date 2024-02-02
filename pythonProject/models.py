import gin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

@gin.configurable
def lstm_model(num_recurrent_layers, num_fc_layers, num_hidden_units, dropout_rate, learning_rate,
               stateful, input_shape, num_classes, batch_size=None):
    model = Sequential()

    for i in range(num_recurrent_layers):
        return_sequences = True if i < num_recurrent_layers - 1 else False

        if stateful:
            assert batch_size is not None, "batch_size must be specified for stateful LSTM"
            batch_input_shape = (batch_size,) + input_shape
            model.add(LSTM(num_hidden_units, return_sequences=return_sequences, stateful=stateful,
                           batch_input_shape=batch_input_shape))
        else:
            # 在非状态保持模式下使用原始的input_shape
            model.add(LSTM(num_hidden_units, return_sequences=return_sequences, input_shape=input_shape))

        model.add(Dropout(dropout_rate))

    for _ in range(num_fc_layers):
        model.add(Dense(num_hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))



    return model







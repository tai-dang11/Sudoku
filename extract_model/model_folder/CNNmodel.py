import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.Conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1))
        self.Conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pooling = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)
        self.flt = Flatten()
        self.dropout2 = Dropout(0.45)
        self.d = Dense(10)


    def call(self, inputs, training=False):
        x = self.Conv1(inputs)
        x = self.Conv2(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = self.flt(x)
        x = self.dropout2(x)
        x = self.d(x)
        return x



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


class MyModel(tf.keras.Model):

    def __init__(self,filters):
        super().__init__()

        self.Conv1 = Conv2D(filters=filters, kernel_size=3, padding='same',input_shape=(9, 9, 1))

        self.Conv2_1 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.Conv2_2 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.Conv2_3 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.Conv2_4 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.Conv2_5 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.Conv2_6 = Conv2D(filters=filters, kernel_size=3, padding='same')

        self.Conv3 = Conv2D(filters=9, kernel_size=1, padding='same')

        self.BatchNorm = tf.keras.layers.BatchNormalization()
        self.BatchNorm2 = tf.keras.layers.BatchNormalization()
        self.BatchNorm3 = tf.keras.layers.BatchNormalization()
        self.BatchNorm4 = tf.keras.layers.BatchNormalization()
        self.BatchNorm5 = tf.keras.layers.BatchNormalization()
        self.BatchNorm6 = tf.keras.layers.BatchNormalization()
        self.BatchNorm7 = tf.keras.layers.BatchNormalization()

        self.ReLU1 = tf.keras.layers.ReLU()
        self.ReLU2 = tf.keras.layers.ReLU()
        self.ReLU3 = tf.keras.layers.ReLU()
        self.ReLU4 = tf.keras.layers.ReLU()
        self.ReLU5 = tf.keras.layers.ReLU()
        self.ReLU6 = tf.keras.layers.ReLU()
        self.ReLU7 = tf.keras.layers.ReLU()

    def call(self, inputs):

        x = self.Conv1(inputs)
        x = self.BatchNorm(x)
        x = self.ReLU1(x)

        x = self.Conv2_1(x)
        x = self.BatchNorm2(x)
        x = self.ReLU2(x)

        x = self.Conv2_2(x)
        x = self.BatchNorm3(x)
        x = self.ReLU3(x)

        x = self.Conv2_3(x)
        x = self.BatchNorm4(x)
        x = self.ReLU4(x)

        x = self.Conv2_4(x)
        x = self.BatchNorm5(x)
        x = self.ReLU5(x)

        x = self.Conv2_5(x)
        x = self.BatchNorm6(x)
        x = self.ReLU6(x)

        x = self.Conv2_6(x)
        x = self.BatchNorm7(x)
        x = self.ReLU7(x)

        x = self.Conv3(x)

        return x




import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from create_data import processed_data
from model import MyModel

# callback function to save checkpoints, and loss/accuracy graphs in Tensorboard
def callback(checkpoint_path, LOGS):

    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, mode='max',save_best_only=True, monitor='sparse_categorical_accuracy')
    reduce_lr = ReduceLROnPlateau(factor=0.50, monitor='sparse_categorical_accuracy', patience=3, min_lr=0.000001, verbose=1)
    tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, reduce_lr, tensorboard]

    return callbacks_list


def train(epochs,batch_size,lr,filters,checkpoint_path, LOGS):
    ds_train, ds_test, steps = processed_data(batch_size)

    callbacks_list = callback(checkpoint_path,LOGS)

    model = MyModel(filters)
    model.build((None, 9, 9, 1))
    model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(
        ds_train,
        epochs = epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data = ds_test,
        steps_per_epoch = steps
    )


checkpoint_path = "/logs/1"
LOGS = "/Users/dttai11/Sudoku-Solver/logs/1/tensorboard"
train(epochs = 18, batch_size= 64, lr = 0.01, filters = 256, checkpoint_path = checkpoint_path, LOGS = LOGS)
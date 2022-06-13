import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from extract_model.model_folder.SHVNdataset import shsvnData
from extract_model.model_folder.CNNmodel import MyModel

def callback(checkpoint_path, LOGS):

    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, mode='min')
    reduce_lr = ReduceLROnPlateau(factor=0.50, monitor='loss', patience=3, min_lr=0.000001, verbose=1)
    tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, reduce_lr, tensorboard]

    return callbacks_list

ds_train, ds_test = shsvnData()

def train(epochs,data_train,data_test,lr):
    checkpoint_path = "/Users/dttai11/Sudoku/backend/api/extract_model/logs"
    LOGS = '/Users/dttai11/Sudoku/backend/api/extract_model/logs/tensorboard1'

    model = MyModel()
    callbacks_list = callback(checkpoint_path,LOGS)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=['accuracy'],
    )

    model.fit(
        data_train,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=data_test
    )


train(epochs = 1, data_train= ds_train, data_test = ds_test,lr = 0.001)
#
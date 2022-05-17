import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import preprocess

answer_path = '/dataset/processed_data/ans.npy'
puzzel_path = '/dataset/processed_data/puzzle.npy'
preprocess(answer_path, puzzel_path)

# normalize image
def normalize(image, y):
    return (tf.cast(image, tf.float32) / 9.) - 0.5, y

# create test set and train set from saved raw data
def processed_data(batch_size):

    puzzle = np.load('../dataset/processed_data/puzzle.npy')
    answer = np.load('../dataset/processed_data/ans.npy')

    x_train, y_train = puzzle[...,np.newaxis], answer[..., np.newaxis] - 1
    X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train,test_size=0.15, shuffle=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE).repeat().batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)

    # steps per epochs for training
    steps = len(x_train) // batch_size

    return train_dataset, test_dataset, steps


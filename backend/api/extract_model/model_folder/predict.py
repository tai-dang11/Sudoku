import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('/Users/dttai11/Sudoku/logs')
sudoku_numbers = np.load('/preprocess_image/nparray/ans.npy')
out = model.predict(sudoku_numbers)
puzzle=np.argmax(out,axis=-1).reshape((9,9))
print(puzzle)